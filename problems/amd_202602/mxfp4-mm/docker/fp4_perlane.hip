// FP4 GEMM with per-lane loading: each lane loads its own row
// Enables LDS swizzle since write-row == read-row
// Trades coalesced global loads for zero bank conflicts
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 32
#define TILE_N 32
#define K_STEP 128
#define NWARPS 4
#define NTHREADS (64 * NWARPS)

extern "C" __global__ __launch_bounds__(NTHREADS, 4)
void fp4_perlane(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    unsigned short* __restrict__ C,
    int M, int N, int K_bytes
) {
    // No LDS needed for A/B data — load directly to registers per lane
    // Only use LDS for the output store (if needed)

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

    // Each lane's row in global memory
    int a_row = arb + lane32;
    int b_col = bcb + lane32;

    // Pre-compute global base pointers for this lane
    const unsigned char* a_base = A + a_row * K_bytes;
    const unsigned char* b_base = B + b_col * K_bytes;
    int a_scale_base = a_row * Kg;
    int b_scale_base = b_col * Kg;

    // Bounds check
    bool a_valid = (a_row < M);
    bool b_valid = (b_col < N);

    for (int ki = 0; ki < KI; ki++) {
        int k_off = ki * K_STEP;

        // Pre-load all 8 scale values for this K-tile
        int sb = ki * (K_STEP / 16);
        unsigned char as8[8], bs8[8];
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            as8[s] = a_valid ? A_scale[a_scale_base + sb + s] : 127;
            bs8[s] = b_valid ? B_scale[b_scale_base + sb + s] : 127;
        }

        // 4 MFMA sub-iterations per K-tile
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int byte_off = k_off + sub * 32 + group * 16;

            // Per-lane load: each lane reads 16 bytes from its own row
            v8i a_reg = {}, b_reg = {};
            if (a_valid) {
                *(uint4*)&a_reg = *(const uint4*)&a_base[byte_off];
            }
            if (b_valid) {
                *(uint4*)&b_reg = *(const uint4*)&b_base[byte_off];
            }

            unsigned char asc = as8[sub * 2 + group];
            unsigned char bsc = bs8[sub * 2 + group];
            unsigned int pas = (unsigned int)asc | ((unsigned int)asc << 8);
            unsigned int pbs = (unsigned int)bsc | ((unsigned int)bsc << 8);

            __builtin_amdgcn_sched_barrier(0);
            c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, c_acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
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
    fp4_perlane<<<total_tiles, NTHREADS, 0, 0>>>(
        (const unsigned char*)A.data_ptr(), (const unsigned char*)B.data_ptr(),
        (const unsigned char*)As.data_ptr(), (const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out.data_ptr(), M_val, (int)N_val, K_val);
    return g_out;
}

TORCH_LIBRARY(hk_gemm, m) {
    m.def("run_lds", &run_fp4_lds);
}
