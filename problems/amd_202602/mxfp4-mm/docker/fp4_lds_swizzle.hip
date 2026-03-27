// FP4 LDS GEMM with XOR swizzle for bank-conflict-free access
// Based on the correct 7.7µs kernel, adding:
// 1. LDS XOR swizzle (row XOR into column address)
// 2. Better barrier placement
// 3. Scale pre-loading remains
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 32
#define TILE_N 32
#define K_STEP 128
#define NWARPS 4
#define NTHREADS (64 * NWARPS)

// XOR swizzle: conflict-free for 32 rows accessing same K-column
// Each bank is 4 bytes. K_STEP=128 bytes = 32 banks worth.
// Swizzle: actual_col = col ^ ((row & 7) << 4)
// This shifts the bank selection by row index, avoiding conflicts
#define SWIZZLE_WRITE(buf, row, col) buf[row][(col) ^ (((row) & 7) << 4)]
#define SWIZZLE_READ(buf, row, col) buf[row][(col) ^ (((row) & 7) << 4)]

extern "C" __global__ __launch_bounds__(NTHREADS, 4)
void fp4_lds_swizzle(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    unsigned short* __restrict__ C,
    int M, int N, int K_bytes
) {
    __shared__ unsigned char lds_a[2][TILE_M][K_STEP + 16];
    __shared__ unsigned char lds_b[2][TILE_N][K_STEP + 16];

    int bid = blockIdx.x;
    int npm = M / TILE_M, npn = N / TILE_N;
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
    v16f c_acc = {};

    int arb = pm * TILE_M, bcb = pn * TILE_N;
    int KI = K_bytes / K_STEP, Kg = K_bytes / 16;

    // Cooperative load with XOR swizzle: first K-tile
    {
        int af = tid * 16, ar = af / K_STEP, ac = af % K_STEP;
        int agr = arb + ar;
        uint4 av = (agr < M) ? *(const uint4*)&A[agr * K_bytes + ac] : make_uint4(0,0,0,0);
        // Swizzle write: XOR col with (row & 7) << 4
        *(uint4*)&lds_a[0][ar][ac ^ ((ar & 7) << 4)] = av;

        int bf = tid * 16, br = bf / K_STEP, bc = bf % K_STEP;
        int bgc = bcb + br;
        uint4 bv = (bgc < N) ? *(const uint4*)&B[bgc * K_bytes + bc] : make_uint4(0,0,0,0);
        *(uint4*)&lds_b[0][br][bc ^ ((br & 7) << 4)] = bv;
    }
    __syncthreads();

    int tic = 0, toc = 1;
    for (int ki = 0; ki < KI; ki++, tic ^= 1, toc ^= 1) {
        // Prefetch next tile with swizzle
        if (ki + 1 < KI) {
            int kn = (ki + 1) * K_STEP;
            int af = tid * 16, ar = af / K_STEP, ac = af % K_STEP;
            int agr = arb + ar;
            uint4 av = (agr < M) ? *(const uint4*)&A[agr * K_bytes + kn + ac] : make_uint4(0,0,0,0);
            *(uint4*)&lds_a[toc][ar][ac ^ ((ar & 7) << 4)] = av;

            int bf = tid * 16, br = bf / K_STEP, bc = bf % K_STEP;
            int bgc = bcb + br;
            uint4 bv = (bgc < N) ? *(const uint4*)&B[bgc * K_bytes + kn + bc] : make_uint4(0,0,0,0);
            *(uint4*)&lds_b[toc][br][bc ^ ((br & 7) << 4)] = bv;
        }

        // Pre-load scales
        int sb2 = ki * (K_STEP / 16);
        int asr = (arb + lane32) * Kg, bsr = (bcb + lane32) * Kg;
        unsigned char as8[8], bs8[8];
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            as8[s] = A_scale[asr + sb2 + s];
            bs8[s] = B_scale[bsr + sb2 + s];
        }

        // Compute with swizzled reads
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int kbs = sub * 32 + group * 16;
            // Swizzle read: same XOR pattern
            int swizzled_col = kbs ^ ((lane32 & 7) << 4);

            v8i a_reg = {}, b_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_a[tic][lane32][swizzled_col];
            *(uint4*)&b_reg = *(uint4*)&lds_b[tic][lane32][swizzled_col];

            unsigned char asc = as8[sub * 2 + group], bsc = bs8[sub * 2 + group];
            unsigned int pas = (unsigned int)asc | ((unsigned int)asc << 8);
            unsigned int pbs = (unsigned int)bsc | ((unsigned int)bsc << 8);

            __builtin_amdgcn_sched_barrier(0);
            c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, c_acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // Store bf16
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = arb + group * 4 + i * 8 + j;
            int c = bcb + lane32;
            if (r < M && c < N) {
                float val = c_acc[i * 4 + j];
                unsigned int u = __float_as_uint(val);
                u = u + (((u >> 16) & 1) + 0x7FFF);
                C[r * N + c] = (unsigned short)(u >> 16);
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
    int total_tiles = (M_val / TILE_M) * ((int)N_val / TILE_N);
    fp4_lds_swizzle<<<total_tiles, NTHREADS, 0, 0>>>(
        (const unsigned char*)A.data_ptr(), (const unsigned char*)B.data_ptr(),
        (const unsigned char*)As.data_ptr(), (const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out.data_ptr(), M_val, (int)N_val, K_val);
    return g_out;
}

TORCH_LIBRARY(hk_gemm, m) {
    m.def("run_lds", &run_fp4_lds);
}
