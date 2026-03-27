// FP4 GEMM with simple LDS tiling (no HK swizzle)
// Cooperative loading from global → raw LDS, then naive MFMA reads from LDS
// This combines the proven naive MFMA correctness with LDS tiling performance
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 32
#define TILE_N 32
#define K_STEP 64   // 64 bytes per K-tile = 128 FP4 elements = 2 MFMA calls
#define NWARPS 4
#define NTHREADS (64 * NWARPS)

// LDS: double-buffered A and B tiles
// A: TILE_M × K_STEP = 32 × 64 = 2048 bytes
// B: TILE_N × K_STEP = 32 × 64 = 2048 bytes
// ×2 for double buffer = 8192 bytes total
__shared__ unsigned char lds[2][2][32][64];  // [buf][A/B][row][k_byte]

constexpr int GM = 256;
constexpr int GN = 3072;
constexpr int GK = 768;  // K/2 packed bytes

extern "C" __global__ __launch_bounds__(NTHREADS, 4)
void fp4_lds_gemm(
    const unsigned char* __restrict__ A,       // [GM, GK] FP4 packed
    const unsigned char* __restrict__ B,       // [GN, GK] FP4 packed
    const unsigned char* __restrict__ A_scale,  // [GM, GK/16] E8M0 raw
    const unsigned char* __restrict__ B_scale,  // [GN, GK/16] E8M0 raw
    unsigned short* __restrict__ C,             // [GM, GN] bf16
    int M, int N, int K_bytes                   // K_bytes = K/2
) {
    // Block assignment
    int bid = blockIdx.x;
    int npm = M / TILE_M;
    int npn = N / TILE_N;
    int pm = bid / npn;
    int pn = bid % npn;

    int tid = threadIdx.x;
    int warp_id = tid / 64;
    int lane = tid % 64;
    int lane32 = lane & 31;
    int group = lane >> 5;

    // Accumulator
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float v16f;
    v16f c_acc = {};

    int a_row_base = pm * TILE_M;
    int b_col_base = pn * TILE_N;
    int K_iters = K_bytes / K_STEP;

    // Cooperative load: first K-tile
    // 256 threads load 2×2048 = 4096 bytes → 16 bytes per thread
    // Load A tile
    {
        int bytes_per_thread = (TILE_M * K_STEP) / NTHREADS;  // 2048/256 = 8
        for (int i = 0; i < bytes_per_thread; i++) {
            int flat = tid * bytes_per_thread + i;
            int r = flat / K_STEP;
            int c = flat % K_STEP;
            int gr = a_row_base + r;
            lds[0][0][r][c] = (gr < M) ? A[gr * K_bytes + c] : 0;
        }
        // Load B tile
        for (int i = 0; i < bytes_per_thread; i++) {
            int flat = tid * bytes_per_thread + i;
            int r = flat / K_STEP;
            int c = flat % K_STEP;
            int gc = b_col_base + r;
            lds[0][1][r][c] = (gc < N) ? B[gc * K_bytes + c] : 0;
        }
    }
    __syncthreads();

    int tic = 0, toc = 1;
    for (int ki = 0; ki < K_iters; ki++, tic ^= 1, toc ^= 1) {
        // Prefetch next K-tile while computing current
        if (ki + 1 < K_iters) {
            int k_next = (ki + 1) * K_STEP;
            int bytes_per_thread = (TILE_M * K_STEP) / NTHREADS;
            for (int i = 0; i < bytes_per_thread; i++) {
                int flat = tid * bytes_per_thread + i;
                int r = flat / K_STEP;
                int c = flat % K_STEP;
                int gr = a_row_base + r;
                lds[toc][0][r][c] = (gr < M) ? A[gr * K_bytes + k_next + c] : 0;
                int gc = b_col_base + r;
                lds[toc][1][r][c] = (gc < N) ? B[gc * K_bytes + k_next + c] : 0;
            }
        }

        // Compute: 2 FP4 MFMA calls per K-tile
        // (64 bytes / 32 bytes per MFMA = 2)
        for (int sub = 0; sub < 2; sub++) {
            // Read from LDS: lane32=row/col, group=K-half within 32 bytes
            int a_row = lane32;
            int k_byte_start = sub * 32 + group * 16;

            unsigned char a_fp4[16];
            #pragma unroll
            for (int i = 0; i < 16; i++)
                a_fp4[i] = lds[tic][0][a_row][k_byte_start + i];

            unsigned char b_fp4[16];
            int b_col = lane32;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                b_fp4[i] = lds[tic][1][b_col][k_byte_start + i];

            v8i a_reg = {};
            a_reg[0] = *(int*)&a_fp4[0]; a_reg[1] = *(int*)&a_fp4[4];
            a_reg[2] = *(int*)&a_fp4[8]; a_reg[3] = *(int*)&a_fp4[12];

            v8i b_reg = {};
            b_reg[0] = *(int*)&b_fp4[0]; b_reg[1] = *(int*)&b_fp4[4];
            b_reg[2] = *(int*)&b_fp4[8]; b_reg[3] = *(int*)&b_fp4[12];

            // E8M0 scale loading: v3 packing (broadcast own to both bytes)
            // Each sub-MFMA processes 32 FP4 values = 1 scale group
            // Scale group index: ki * (K_STEP/16) + sub * 2 + group
            int K_groups = K_bytes / 16;  // K/32 scale groups
            int scale_idx = ki * (K_STEP / 16) + sub * 2 + group;
            int a_row_for_scale = a_row_base + lane32;
            int b_col_for_scale = b_col_base + lane32;
            unsigned char a_sc = (a_row_for_scale < M && scale_idx < K_groups) ?
                A_scale[a_row_for_scale * K_groups + scale_idx] : 127;
            unsigned char b_sc = (b_col_for_scale < N && scale_idx < K_groups) ?
                B_scale[b_col_for_scale * K_groups + scale_idx] : 127;
            unsigned int pas = (unsigned int)a_sc | ((unsigned int)a_sc << 8);
            unsigned int pbs = (unsigned int)b_sc | ((unsigned int)b_sc << 8);

            c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, c_acc, 4, 4, 0, pas, 0, pbs);
        }

        __syncthreads();  // ensure prefetch complete before next iteration
    }

    // Store bf16 output
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int r = a_row_base + group * 4 + i * 8 + j;
            int c = b_col_base + lane32;
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
                           torch::Tensor A_scale, torch::Tensor B_scale, int64_t N_val) {
    int M_val = (int)A.size(0), K_val = (int)A.size(1);
    if (M_val != g_m || (int)N_val != g_n) {
        g_out = torch::empty({M_val, N_val}, torch::dtype(torch::kBFloat16).device(A.device()));
        g_m = M_val; g_n = (int)N_val;
    }
    int total_tiles = (M_val / TILE_M) * ((int)N_val / TILE_N);
    fp4_lds_gemm<<<total_tiles, NTHREADS, 0, 0>>>(
        (const unsigned char*)A.data_ptr(),
        (const unsigned char*)B.data_ptr(),
        (const unsigned char*)A_scale.data_ptr(),
        (const unsigned char*)B_scale.data_ptr(),
        (unsigned short*)g_out.data_ptr(),
        M_val, (int)N_val, K_val);
    return g_out;
}

TORCH_LIBRARY(hk_gemm, m) {
    m.def("run_lds", &run_fp4_lds);
}
