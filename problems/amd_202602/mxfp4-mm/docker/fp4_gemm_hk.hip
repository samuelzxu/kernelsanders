// FP4 fused quant+GEMM using HipKittens primitives
// Based on FP8 4-wave kernel structure, adapted for FP4 with inline A quant
// Compiles with: hipcc -O3 -std=c++20 --offload-arch=gfx950 -DKITTENS_CDNA4 -I<hk>/include

#include "kittens.cuh"
using namespace kittens;

// Configuration for K=1536 M=256 N=3072
// Tile: 128M x 128N output, K_STEP=128 (= 64 FP4 pairs = 128 FP4 elements per MFMA)
// 4 wavefronts, each handles 64x64 sub-tile
// 2x2 sub-tile layout per block

// For now: use the FP8 tile types for storage (fp8e4m3_4 = 4 bytes = 8 FP4 values)
// The MFMA instruction treats the same bits as FP4 when type=4

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 128;  // 128 FP4 elements = 64 packed bytes
#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

// Shared tile types (reuse FP8 tiles for FP4 packed storage)
using ST_A = st_fp8e4m3<BLOCK_M / 2, BLOCK_K, st_16x128_s>;
using ST_B = st_fp8e4m3<BLOCK_N / 2, BLOCK_K, st_16x128_s>;

// Register tile types
using RT_A = rt_fp8e4m3<BLOCK_M / 2 / 2, BLOCK_K>; // 32x128
using RT_B = rt_fp8e4m3<BLOCK_N / 2 / 2, BLOCK_K>; // 32x128
using RT_C = rt_fl<BLOCK_M / 2 / 2, BLOCK_N / 2 / 2, col_l, rt_16x16_s>; // 32x32 f32

// Global layout types (FP8 storage but contains FP4 data)
using GL_A = kittens::gl<fp8e4m3, 1, 1, -1, -1>;  // variable M, K
using GL_B = kittens::gl<fp8e4m3, 1, 1, -1, -1>;  // variable N, K
using GL_C = kittens::gl<bf16, 1, 1, -1, -1>;      // variable M, N

struct fp4_gemm_globals {
    GL_A a;
    GL_B b;
    GL_C c;
    int M, N, K;
};

extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void fp4_gemm_hk(const fp4_gemm_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    int warp_id = kittens::warpid();
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    RT_C c;
    zero(c);

    // Block assignment with XCD-aware swizzling
    int wgid = blockIdx.x;
    const int NUM_WGS = gridDim.x;
    const int NUM_XCDS = 8;
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    const int WGM = 4;
    int num_pid_m = (g.M + BLOCK_M - 1) / BLOCK_M;
    int num_pid_n = (g.N + BLOCK_N - 1) / BLOCK_N;
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;

    int block_row = pid_m;
    int block_col = pid_n;
    int k_iters = g.K / BLOCK_K;

    using G = kittens::group<NUM_WARPS>;

    // Load first K-tile
    G::load(As[0], g.a, {0, 0, block_row, 0});
    G::load(Bs[0], g.b, {0, 0, block_col, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    RT_A a_tile;
    RT_B b_tile;

    int tic = 0, toc = 1;
    for (int k = 0; k < k_iters; k++, tic ^= 1, toc ^= 1) {
        // Load A and B sub-tiles from shared memory
        load(a_tile, subtile_inplace<BLOCK_M/2/2, BLOCK_K>(As[tic], {warp_row, 0}));
        load(b_tile, subtile_inplace<BLOCK_N/2/2, BLOCK_K>(Bs[tic], {warp_col, 0}));

        // Prefetch next K-tile
        if (k + 1 < k_iters) {
            G::load(As[toc], g.a, {0, 0, block_row, k + 1});
            G::load(Bs[toc], g.b, {0, 0, block_col, k + 1});
        }
        __builtin_amdgcn_s_barrier();

        // MFMA: use FP8 MFMA (treats FP4-packed data as FP8 for now)
        // TODO: replace with FP4 scaled MFMA when scale loading is added
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c, a_tile, b_tile, c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Store result: convert f32 accumulators to bf16 and write to global
    // c register tile has layout (rows, cols) relative to the warp's sub-tile
    // Each warp handles BLOCK_M/4 x BLOCK_N/4 = 32x32 sub-tile
    // TODO: use HK store primitive when working, for now just mark compiled
}

// Simple test entry point
int main() {
    return 0;
}
