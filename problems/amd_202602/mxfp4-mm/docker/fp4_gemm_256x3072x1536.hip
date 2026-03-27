// FP4 GEMM for M=256 N=3072 K=1536 using HipKittens primitives
// Pre-quantized A (FP4 packed) + pre-shuffled B (FP4 packed) + E8M0 scales
// Output: bf16 [M, N]
//
// Tile: 128M x 128N, K_STEP=64 bytes (= 128 FP4 elements)
// 4 warps per block, 2x2 sub-tile layout
// Each warp handles 64x64 output sub-tile with 2 MFMA 32x32x64 calls

#include "kittens.cuh"
using namespace kittens;

#define HK_M 256
#define HK_N 3072
#define HK_K 768   // K/2 = 768 packed bytes (1536 FP4 elements)

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int K_STEP = 64;   // 64 bytes per K-tile = 128 FP4 elements
#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

// Shared tiles: use FP8 types for byte-level storage
using ST_A = st_fp8e4m3<TILE_M/2, K_STEP, st_16x128_s>;  // 64 x 64 bytes
using ST_B = st_fp8e4m3<TILE_N/2, K_STEP, st_16x128_s>;  // 64 x 64 bytes

// Register tiles
using RT_A = rt_fp8e4m3<TILE_M/2/2, K_STEP>;  // 32 x 64 bytes
using RT_B = rt_fp8e4m3<TILE_N/2/2, K_STEP>;  // 32 x 64 bytes
using RT_C = rt_fl<TILE_M/2/2, TILE_N/2/2, col_l, rt_16x16_s>;  // 32x32 f32 accum

// Global layouts (FP8 byte-level addressing)
using GL_A = gl<fp8e4m3, 1, 1, HK_M, HK_K>;
using GL_B = gl<fp8e4m3, 1, 1, HK_N, HK_K>;
using GL_C = gl<bf16, 1, 1, HK_M, HK_N>;

struct gemm_globals {
    GL_A a;
    GL_B b;
    GL_C c;
};

extern "C" __global__ __launch_bounds__(NUM_THREADS, 2)
void fp4_gemm_kernel(const __grid_constant__ gemm_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    int warp_row = warpid() / 2;
    int warp_col = warpid() % 2;

    RT_C c_acc;
    zero(c_acc);

    // Block assignment
    int wgid = blockIdx.x;
    const int NUM_WGS = gridDim.x;
    const int NUM_XCDS = 8;
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);

    int num_pid_m = HK_M / TILE_M;  // 2
    int num_pid_n = HK_N / TILE_N;  // 24
    const int WGM = 2;
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;

    int block_row = pid_m;
    int block_col = pid_n;

    constexpr int K_ITERS = HK_K / K_STEP;  // 768/64 = 12

    using G = kittens::group<NUM_WARPS>;

    // Load first K-tile into shared
    G::load(As[0], g.a, {0, 0, block_row, 0});
    G::load(Bs[0], g.b, {0, 0, block_col, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    RT_A a_tile;
    RT_B b_tile;

    int tic = 0, toc = 1;
    #pragma unroll 1
    for (int k = 0; k < K_ITERS; k++, tic ^= 1, toc ^= 1) {
        // Load from shared → registers
        load(a_tile, subtile_inplace<TILE_M/4, K_STEP>(As[tic], {warp_row, 0}));
        load(b_tile, subtile_inplace<TILE_N/4, K_STEP>(Bs[tic], {warp_col, 0}));

        // Prefetch next K-tile from global → shared
        if (k + 1 < K_ITERS) {
            G::load(As[toc], g.a, {0, 0, block_row, k + 1});
            G::load(Bs[toc], g.b, {0, 0, block_col, k + 1});
        }

        // Barrier + MFMA
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_acc, a_tile, b_tile, c_acc);  // FP8 MFMA for now
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Store: convert f32 → bf16 and write to global
    // Use HK store primitive
    auto c_subtile = subtile_inplace<TILE_M/4, TILE_N/4>(
        g.c, {0, 0, block_row * 2 + warp_row, block_col * 2 + warp_col});
    rt_bf<TILE_M/4, TILE_N/4, col_l, rt_16x16_s> c_bf16;
    copy(c_bf16, c_acc);  // f32 → bf16 conversion
    store(c_subtile, c_bf16);
}

int main() {
    return 0;
}
