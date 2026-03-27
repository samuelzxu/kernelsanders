// FP4 GEMM with 2 N-tiles per block for latency hiding
// Each block produces 32×64 output (2 × 32×32 tiles)
// MFMA for tile 0 overlaps with LDS loads for tile 1
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 32
#define TILE_N 32
#define K_STEP 128
#define N_TILES 2    // 2 N-tiles per block
#define NWARPS 4
#define NTHREADS (64 * NWARPS)

extern "C" __global__ __launch_bounds__(NTHREADS, 2)
void fp4_2tile(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    unsigned short* __restrict__ C,
    int M, int N, int K_bytes
) {
    // LDS: A shared between tiles, B separate per tile
    __shared__ unsigned char lds_a[2][TILE_M][K_STEP + 16];
    __shared__ unsigned char lds_b[N_TILES][2][TILE_N][K_STEP + 16];

    int bid = blockIdx.x;
    int npm = M / TILE_M, npn = N / (TILE_N * N_TILES);
    int NW = gridDim.x, NXCD = 8;
    bid = (bid % NXCD) * (NW / NXCD) + (bid / NXCD);
    int WGM = (npm < 8) ? npm : 8;
    int nwig = WGM * npn;
    int gid = bid / nwig, fpm = gid * WGM;
    int gsm = min(npm - fpm, WGM);
    int pm = fpm + ((bid % nwig) % gsm);
    int pn = (bid % nwig) / gsm;

    int tid = threadIdx.x;
    int lane = tid % 64, lane32 = lane & 31, group = lane >> 5;

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float v16f;
    v16f c_acc[N_TILES];
    for (int t = 0; t < N_TILES; t++) c_acc[t] = {};

    int arb = pm * TILE_M;
    int bcb[N_TILES];
    for (int t = 0; t < N_TILES; t++) bcb[t] = (pn * N_TILES + t) * TILE_N;

    int KI = K_bytes / K_STEP, Kg = K_bytes / 16;

    // Load first K-tile: A (shared) + B for both tiles
    {
        // A: 128 threads load A tile (32×128 = 4096 bytes, 16 bytes/thread for 256 total)
        // Split: first 128 threads load A, next 128 split between B tiles
        if (tid < 128) {
            // Load A
            int a_flat = tid * (TILE_M * K_STEP / 128);  // 32 bytes per thread
            for (int ld = 0; ld < 2; ld++) {
                int af = tid * 16 + ld * 128 * 16;
                if (af < TILE_M * K_STEP) {
                    int ar = af / K_STEP, ac = af % K_STEP;
                    int agr = arb + ar;
                    *(uint4*)&lds_a[0][ar][ac] = (agr < M) ?
                        *(const uint4*)&A[agr * K_bytes + ac] : make_uint4(0,0,0,0);
                }
            }
        }
        // B tiles: 256 threads load 2 × 4096 bytes = 8192 total = 32 bytes/thread
        for (int t = 0; t < N_TILES; t++) {
            int bf = tid * 16;
            int br = bf / K_STEP, bc2 = bf % K_STEP;
            if (br < TILE_N) {
                int bgc = bcb[t] + br;
                *(uint4*)&lds_b[t][0][br][bc2] = (bgc < N) ?
                    *(const uint4*)&B[bgc * K_bytes + bc2] : make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    int tic = 0, toc = 1;
    for (int ki = 0; ki < KI; ki++, tic ^= 1, toc ^= 1) {
        // Prefetch next K-tile
        if (ki + 1 < KI) {
            int kn = (ki + 1) * K_STEP;
            if (tid < 128) {
                for (int ld = 0; ld < 2; ld++) {
                    int af = tid * 16 + ld * 128 * 16;
                    if (af < TILE_M * K_STEP) {
                        int ar = af / K_STEP, ac = af % K_STEP;
                        int agr = arb + ar;
                        *(uint4*)&lds_a[toc][ar][ac] = (agr < M) ?
                            *(const uint4*)&A[agr * K_bytes + kn + ac] : make_uint4(0,0,0,0);
                    }
                }
            }
            for (int t = 0; t < N_TILES; t++) {
                int bf = tid * 16;
                int br = bf / K_STEP, bc2 = bf % K_STEP;
                if (br < TILE_N) {
                    int bgc = bcb[t] + br;
                    *(uint4*)&lds_b[t][toc][br][bc2] = (bgc < N) ?
                        *(const uint4*)&B[bgc * K_bytes + kn + bc2] : make_uint4(0,0,0,0);
                }
            }
        }

        // Pre-load scales for both tiles
        int sb2 = ki * (K_STEP / 16);
        int asr = (arb + lane32) * Kg;
        unsigned char as8[8];
        #pragma unroll
        for (int s = 0; s < 8; s++) as8[s] = A_scale[asr + sb2 + s];

        unsigned char bs8[N_TILES][8];
        for (int t = 0; t < N_TILES; t++) {
            int bsr = (bcb[t] + lane32) * Kg;
            #pragma unroll
            for (int s = 0; s < 8; s++) bs8[t][s] = B_scale[bsr + sb2 + s];
        }

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);

        // INTERLEAVED MFMA: alternate between tile 0 and tile 1
        // This hides LDS read latency for one tile behind MFMA for the other
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int kbs = sub * 32 + group * 16;

            // Load A (shared between tiles)
            v8i a_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_a[tic][lane32][kbs];

            // Tile 0: load B, compute MFMA
            v8i b0 = {};
            *(uint4*)&b0 = *(uint4*)&lds_b[0][tic][lane32][kbs];
            unsigned int pas0 = (unsigned int)as8[sub*2+group] | ((unsigned int)as8[sub*2+group] << 8);
            unsigned int pbs0 = (unsigned int)bs8[0][sub*2+group] | ((unsigned int)bs8[0][sub*2+group] << 8);
            __builtin_amdgcn_sched_barrier(0);
            c_acc[0] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b0, c_acc[0], 4, 4, 0, pas0, 0, pbs0);

            // Tile 1: load B, compute MFMA (overlaps with tile 0's MFMA retirement)
            v8i b1 = {};
            *(uint4*)&b1 = *(uint4*)&lds_b[1][tic][lane32][kbs];
            unsigned int pbs1 = (unsigned int)bs8[1][sub*2+group] | ((unsigned int)bs8[1][sub*2+group] << 8);
            __builtin_amdgcn_sched_barrier(0);
            c_acc[1] = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b1, c_acc[1], 4, 4, 0, pas0, 0, pbs1);
            __builtin_amdgcn_sched_barrier(0);
        }

        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // Store both tiles
    for (int t = 0; t < N_TILES; t++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int r = arb + group * 4 + i * 8 + j;
                int c = bcb[t] + lane32;
                if (r < M && c < N) {
                    float val = c_acc[t][i * 4 + j];
                    unsigned int u = __float_as_uint(val);
                    u = u + (((u >> 16) & 1) + 0x7FFF);
                    C[r * N + c] = (unsigned short)(u >> 16);
                }
            }
        }
    }
}

static torch::Tensor g_out;
static int g_m = 0, g_n = 0;

torch::Tensor run_fp4_lds(torch::Tensor A, torch::Tensor B,
                           torch::Tensor As, torch::Tensor Bs, int64_t N_val) {
    int M_val = (int)A.size(0), K_val = (int)A.size(1);
    if (M_val != g_m || (int)N_val != g_n) {
        g_out = torch::empty({M_val, N_val}, torch::dtype(torch::kBFloat16).device(A.device()));
        g_m = M_val; g_n = (int)N_val;
    }
    // Grid: half the blocks since each processes 2 N-tiles
    int total_tiles = (M_val / TILE_M) * ((int)N_val / (TILE_N * N_TILES));
    fp4_2tile<<<total_tiles, NTHREADS, 0, 0>>>(
        (const unsigned char*)A.data_ptr(), (const unsigned char*)B.data_ptr(),
        (const unsigned char*)As.data_ptr(), (const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out.data_ptr(), M_val, (int)N_val, K_val);
    return g_out;
}

TORCH_LIBRARY(hk_gemm, m) {
    m.def("run_lds", &run_fp4_lds);
}
