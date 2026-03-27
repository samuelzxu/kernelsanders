// Optimized FP4 GEMM: vectorized loading, K_STEP=128, 4 independent warps
#include <hip/hip_runtime.h>

#define TILE_M 32
#define TILE_N 32
#define K_STEP 128   // 128 bytes = 256 FP4 = 4 MFMA calls
#define NWARPS 4
#define NTHREADS (64 * NWARPS)

// LDS: just 1 set of A/B tiles (no double buffer for simplicity)
// A: 32 × 128 = 4096 bytes, B: 32 × 128 = 4096 bytes
// Total: 8192 bytes — fits easily

extern "C" __global__ __launch_bounds__(NTHREADS, 4)
void fp4_lds_opt(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    unsigned short* __restrict__ C,
    int M, int N, int K_bytes
) {
    // Double-buffered LDS with 16-byte padding per row to reduce bank conflicts
    __shared__ unsigned char lds_a[2][TILE_M][K_STEP + 16];
    __shared__ unsigned char lds_b[2][TILE_N][K_STEP + 16];

    // XCD-aware block assignment (8 XCDs on MI355X)
    int bid = blockIdx.x;
    int npm = M / TILE_M;
    int npn = N / TILE_N;
    int NW = gridDim.x;
    int NXCD = 8;
    bid = (bid % NXCD) * (NW / NXCD) + (bid / NXCD);
    int WGM = (npm < 8) ? npm : 8;
    int nwig = WGM * npn;
    int gid = bid / nwig;
    int fpm = gid * WGM;
    int gsm = min(npm - fpm, WGM);
    int pm = fpm + ((bid % nwig) % gsm);
    int pn = (bid % nwig) / gsm;

    int tid = threadIdx.x;
    int lane = tid % 64;
    int lane32 = lane & 31;
    int group = lane >> 5;

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float v16f;
    v16f c_acc = {};

    int a_row_base = pm * TILE_M;
    int b_col_base = pn * TILE_N;
    int K_iters = K_bytes / K_STEP;
    int K_groups = K_bytes / 16;  // K/32 scale groups

    // Load first K-tile into buffer 0
    {
        int a_flat = tid * 16;
        int a_r = a_flat / K_STEP, a_c = a_flat % K_STEP;
        int a_gr = a_row_base + a_r;
        *(uint4*)&lds_a[0][a_r][a_c] = (a_gr < M) ?
            *(const uint4*)&A[a_gr * K_bytes + a_c] : make_uint4(0,0,0,0);
        int b_flat = tid * 16;
        int b_r = b_flat / K_STEP, b_c = b_flat % K_STEP;
        int b_gc = b_col_base + b_r;
        *(uint4*)&lds_b[0][b_r][b_c] = (b_gc < N) ?
            *(const uint4*)&B[b_gc * K_bytes + b_c] : make_uint4(0,0,0,0);
    }
    __syncthreads();

    int tic = 0, toc = 1;
    for (int ki = 0; ki < K_iters; ki++, tic ^= 1, toc ^= 1) {
        // Prefetch NEXT K-tile into alternate buffer (overlap with compute)
        if (ki + 1 < K_iters) {
            int k_next = (ki + 1) * K_STEP;
            int a_flat = tid * 16;
            int a_r = a_flat / K_STEP, a_c = a_flat % K_STEP;
            int a_gr = a_row_base + a_r;
            *(uint4*)&lds_a[toc][a_r][a_c] = (a_gr < M) ?
                *(const uint4*)&A[a_gr * K_bytes + k_next + a_c] : make_uint4(0,0,0,0);
            int b_flat = tid * 16;
            int b_r = b_flat / K_STEP, b_c = b_flat % K_STEP;
            int b_gc = b_col_base + b_r;
            *(uint4*)&lds_b[toc][b_r][b_c] = (b_gc < N) ?
                *(const uint4*)&B[b_gc * K_bytes + k_next + b_c] : make_uint4(0,0,0,0);
        }

        // Preload all 8 scale values for this K-tile
        int scale_base = ki * (K_STEP / 16);
        int a_scale_row = (a_row_base + lane32) * K_groups;
        int b_scale_row = (b_col_base + lane32) * K_groups;
        unsigned char a_scales[8], b_scales[8];
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            a_scales[s] = A_scale[a_scale_row + scale_base + s];
            b_scales[s] = B_scale[b_scale_row + scale_base + s];
        }

        // Compute 4 FP4 MFMA with pre-loaded scales
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int k_byte_start = sub * 32 + group * 16;

            v8i a_reg = {}, b_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_a[tic][lane32][k_byte_start];
            *(uint4*)&b_reg = *(uint4*)&lds_b[tic][lane32][k_byte_start];

            unsigned char a_sc = a_scales[sub * 2 + group];
            unsigned char b_sc = b_scales[sub * 2 + group];
int main() { return 0; }
