#include "/hk/include/kittens.cuh"
using namespace kittens;
#include "profile_utils.cpp"
#include "fp8_utils.cpp"

// Include the 4-wave kernel (has hardcoded 8192x8192 dims)
#define SIZE 8192

// Forward-declare the kernel template from 4_wave.cu
// Actually, let me just build a simple benchmark

#define M 256
#define N 3072
#define K 1536

// Use the FP8 types directly for a simple test kernel
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;
#define NWARPS 4
#define NTHREADS (kittens::WARP_THREADS * NWARPS)

using ST_A_type = st_fp8e4m3<TILE_M, TILE_K, st_16x128_s>;
using ST_B_type = st_fp8e4m3<TILE_N, TILE_K, st_16x128_s>;
using GL_A_type = gl<fp8e4m3, 1, 1, M, K/2>;
using GL_B_type = gl<fp8e4m3, 1, 1, N, K/2>;
using GL_C_type = gl<bf16, 1, 1, M, N>;

struct test_globals {
    GL_A_type a;
    GL_B_type b;
    GL_C_type c;
};

extern "C" __global__ __launch_bounds__(NTHREADS, 2)
void fp8_test_gemm(const __grid_constant__ test_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A_type (&As)[2] = al.allocate<ST_A_type, 2>();
    ST_B_type (&Bs)[2] = al.allocate<ST_B_type, 2>();

    using G = kittens::group<NWARPS>;

    // Simple: just load tiles to verify it works
    G::load(As[0], g.a, {0, 0, 0, 0});
    G::load(Bs[0], g.b, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
}

int main() { return 0; }
