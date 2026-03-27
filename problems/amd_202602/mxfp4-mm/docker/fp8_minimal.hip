#include "/hk/include/kittens.cuh"
using namespace kittens;

#define HK_M 256
#define HK_N 3072
#define HK_K 768

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;
#define NWARPS 4
#define NTHREADS (kittens::WARP_THREADS * NWARPS)

using ST_A_type = st_fp8e4m3<TILE_M, TILE_K, st_16x128_s>;
using ST_B_type = st_fp8e4m3<TILE_N, TILE_K, st_16x128_s>;
using GL_A_type = gl<fp8e4m3, 1, 1, HK_M, HK_K>;
using GL_B_type = gl<fp8e4m3, 1, 1, HK_N, HK_K>;

struct test_globals {
    GL_A_type a;
    GL_B_type b;
};

extern "C" __global__ __launch_bounds__(NTHREADS, 2)
void hk_load_test(const __grid_constant__ test_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    ST_A_type (&As)[2] = al.allocate<ST_A_type, 2>();
    ST_B_type (&Bs)[2] = al.allocate<ST_B_type, 2>();

    using G = kittens::group<NWARPS>;
    G::load(As[0], g.a, {0, 0, 0, 0});
    G::load(Bs[0], g.b, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    
    // Load subtile to registers
    using RT_A_type = rt_fp8e4m3<TILE_M/2, TILE_K>;
    RT_A_type a_reg;
    load(a_reg, subtile_inplace<TILE_M/2, TILE_K>(As[0], {0, 0}));
}

int main() { return 0; }
