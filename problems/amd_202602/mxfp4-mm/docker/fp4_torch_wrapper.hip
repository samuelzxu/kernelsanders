#include "/hk/include/kittens.cuh"
using namespace kittens;
#include <torch/extension.h>

// HK FP8 GEMM for M=256 N=3072 K=768
// Uses FP8 MFMA (mma_ABt) — to validate tile management correctness
// Then swap to FP4 MFMA once correctness is confirmed
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 128;
#define NWARPS 4
#define NTHREADS (kittens::WARP_THREADS * NWARPS)

using ST_A = st_fp8e4m3<TILE_M, TILE_K, st_16x128_s>;
using ST_B = st_fp8e4m3<TILE_N, TILE_K, st_16x128_s>;
using RT_A = rt_fp8e4m3<TILE_M, TILE_K>;
using RT_B = rt_fp8e4m3<TILE_N, TILE_K>;
using RT_C = rt_fl<TILE_M, TILE_N, col_l, rt_16x16_s>;

constexpr int GM = 256;
constexpr int GN = 3072;
constexpr int GK = 768;

using GL_A = gl<fp8e4m3, 1, 1, GM, GK>;
using GL_B = gl<fp8e4m3, 1, 1, GN, GK>;
using GL_C = gl<bf16, 1, 1, GM, GN>;

struct GG {
    GL_A a;
    GL_B b;
    GL_C c;
};

extern "C" __global__ __launch_bounds__(NTHREADS, 2)
void hk_fp8_gemm(const GG g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A (&As)[2] = al.allocate<ST_A, 2>();
    ST_B (&Bs)[2] = al.allocate<ST_B, 2>();

    using G = kittens::group<NWARPS>;

    int wgid = blockIdx.x;
    constexpr int NXCD = 8;
    wgid = (wgid % NXCD) * (gridDim.x / NXCD) + (wgid / NXCD);
    constexpr int npm = GM / TILE_M;
    constexpr int npn = GN / TILE_N;
    constexpr int WGM = (npm < 8) ? npm : 8;
    constexpr int nwig = WGM * npn;
    int gid = wgid / nwig;
    int fpm = gid * WGM;
    int gsm = min(npm - fpm, WGM);
    int pm = fpm + ((wgid % nwig) % gsm);
    int pn = (wgid % nwig) / gsm;

    RT_C c_acc;
    zero(c_acc);
    RT_A a_reg;
    RT_B b_reg;

    G::load(As[0], g.a, {0, 0, pm, 0});
    G::load(Bs[0], g.b, {0, 0, pn, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    constexpr int KI = GK / TILE_K;
    int tic = 0, toc = 1;
    #pragma unroll 1
    for (int k = 0; k < KI; k++, tic ^= 1, toc ^= 1) {
        load(a_reg, As[tic]);
        load(b_reg, Bs[tic]);
        if (k + 1 < KI) {
            G::load(As[toc], g.a, {0, 0, pm, k + 1});
            G::load(Bs[toc], g.b, {0, 0, pn, k + 1});
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_s_setprio(1);
        // FP4 MFMA with neutral scales (v3 packing)
        // mma_ABt_fp4 does 2 MFMA calls per sub-tile (lo+hi 16 bytes)
        unsigned int scale = 127 | (127 << 8);
        mma_ABt_fp4(c_acc, a_reg, b_reg, c_acc, scale, scale);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
    }

    // Store via LDS
    __builtin_amdgcn_s_barrier();
    {
        shared_allocator al2((int*)&__shm[0]);
        st_bf<TILE_M, TILE_N, st_32x32_s> (&Cs) = al2.allocate<st_bf<TILE_M, TILE_N, st_32x32_s>>();
        rt_bf<TILE_M, TILE_N, col_l, rt_16x16_s> c_bf16;
        copy(c_bf16, c_acc);
        store(Cs, c_bf16);
        __builtin_amdgcn_s_barrier();
        G::store(g.c, Cs, {0, 0, pm, pn});
    }
}

torch::Tensor run_hk_gemm(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({GM, GN}, torch::dtype(torch::kBFloat16).device(A.device()));
    GG globals {
        GL_A((fp8e4m3*)A.data_ptr(), nullptr, nullptr, nullptr, nullptr),
        GL_B((fp8e4m3*)B.data_ptr(), nullptr, nullptr, nullptr, nullptr),
        GL_C((bf16*)C.data_ptr(), nullptr, nullptr, nullptr, nullptr)
    };
    constexpr int total_tiles = (GM / TILE_M) * (GN / TILE_N);
    hk_fp8_gemm<<<total_tiles, NTHREADS, 65536, 0>>>(globals);
    return C;
}

TORCH_LIBRARY(hk_gemm, m) {
    m.def("run", &run_hk_gemm);
}
