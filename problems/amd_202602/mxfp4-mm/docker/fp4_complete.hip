// Complete FP4 GEMM: raw pointer interface, HipKittens LDS+MMA primitives
// For M=256 N=3072 K=1536 (but works for any M/N divisible by 64, K by 256)
// Uses FP8 tile types to store FP4 packed data (same bytes, different interpretation)
// The MFMA currently uses FP8 type (type=0). Will change to FP4 (type=4) after validation.

#include "kittens.cuh"
using namespace kittens;

constexpr int TILE_M = 64;   // M-tile per block
constexpr int TILE_N = 64;   // N-tile per block
constexpr int TILE_K = 128;  // K-bytes per iteration (= 256 FP4 or 128 FP8 elements)
#define NWARPS 4
#define NTHREADS (kittens::WARP_THREADS * NWARPS)

// Shared memory tiles (FP8-typed but storing FP4 packed data)
using ST_A = st_fp8e4m3<TILE_M, TILE_K, st_16x128_s>;
using ST_B = st_fp8e4m3<TILE_N, TILE_K, st_16x128_s>;

// Register tiles
using RT_A = rt_fp8e4m3<TILE_M/2, TILE_K>;  // 32x128
using RT_B = rt_fp8e4m3<TILE_N/2, TILE_K>;  // 32x128
using RT_C = rt_fl<TILE_M/2, TILE_N/2, col_l, rt_16x16_s>;  // 32x32 f32

// Global layout with compile-time M,K dimensions
// We'll template this for different shapes
template<int GM, int GK>
using GL_FP8 = gl<fp8e4m3, 1, 1, GM, GK>;

template<int GM, int GN>
using GL_BF16 = gl<bf16, 1, 1, GM, GN>;

template<int GM, int GN, int GK>
struct GemmGlobals {
    GL_FP8<GM, GK> a;
    GL_FP8<GN, GK> b;
    GL_BF16<GM, GN> c;
};

template<int GM, int GN, int GK>
__global__ __launch_bounds__(NTHREADS, 2)
void gemm_kernel(const GemmGlobals<GM, GN, GK> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    using G = kittens::group<NWARPS>;

    // Block assignment with XCD swizzle
    int wgid = blockIdx.x;
    const int NW = gridDim.x;
    const int NXCD = 8;
    wgid = (wgid % NXCD) * (NW / NXCD) + (wgid / NXCD);
    int npm = GM / TILE_M;
    int npn = GN / TILE_N;
    const int WGM = min(npm, 4);
    int nwig = WGM * npn;
    int gid = wgid / nwig;
    int fpm = gid * WGM;
    int gsm = min(npm - fpm, WGM);
    int pm = fpm + ((wgid % nwig) % gsm);
    int pn = (wgid % nwig) / gsm;

    constexpr int K_ITERS = GK / TILE_K;

    // Load first K-tile
    G::load(As[0], g.a, {0, 0, pm, 0});
    G::load(Bs[0], g.b, {0, 0, pn, 0});

    RT_C c_acc;
    zero(c_acc);
    RT_A a_reg;
    RT_B b_reg;

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    int tic = 0, toc = 1;
    #pragma unroll 1
    for (int k = 0; k < K_ITERS; k++, tic ^= 1, toc ^= 1) {
        // Load sub-tiles from LDS to registers
        load(a_reg, subtile_inplace<TILE_M/2, TILE_K>(As[tic], {0, 0}));
        load(b_reg, subtile_inplace<TILE_N/2, TILE_K>(Bs[tic], {0, 0}));

        // Prefetch next K-tile
        if (k + 1 < K_ITERS) {
            G::load(As[toc], g.a, {0, 0, pm, k + 1});
            G::load(Bs[toc], g.b, {0, 0, pn, k + 1});
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(c_acc, a_reg, b_reg, c_acc);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Store: convert f32 → bf16 and write to global
    rt_bf<TILE_M/2, TILE_N/2, col_l, rt_16x16_s> c_bf16;
    copy(c_bf16, c_acc);
    // For now, skip store - just verify compilation
    // G::store(g.c, c_bf16, {0, 0, pm, pn});
}

// Explicit instantiation
using Globals_256_3072_768 = GemmGlobals<256, 3072, 768>;
template __global__ void gemm_kernel<256, 3072, 768>(const Globals_256_3072_768);

int main() { return 0; }
