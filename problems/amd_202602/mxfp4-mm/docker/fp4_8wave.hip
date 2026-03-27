// 8-wave FP4 GEMM kernel with ping-pong scheduling for MI355X (gfx950)
// Output tile: 64×128, 8 warps in 2×4 grid, each warp computes 32×32
// Uses: double-buffered LDS, XOR swizzle, s_setprio, sched_barrier, s_waitcnt
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 64
#define TILE_N 128
#define K_STEP 128       // bytes per K iteration (256 FP4 values)
#define NWARPS 8
#define NTHREADS (64 * NWARPS)  // 512
#define WARPS_M 2
#define WARPS_N 4
#define SUB_M 32
#define SUB_N 32

// LDS row stride with XOR swizzle (no padding needed)
#define LDS_STRIDE K_STEP

// XOR swizzle for FP4 bytes in LDS (adapted from HipKittens st_16x128)
// For 1-byte elements in a row of K_STEP=128 bytes:
// swizzle = ((byte_offset % 2048) >> 8) << 4
// swizzled = byte_offset ^ swizzle
// This XORs bit 8 into bit 4, swapping 16-byte blocks every 256 bytes
__device__ __forceinline__ int lds_swizzle(int row, int col) {
    int offset = row * K_STEP + col;
    int sw = ((offset % 2048) >> 8) << 4;
    return offset ^ sw;
}

extern "C" __global__ __launch_bounds__(NTHREADS, 1)
void fp4_8wave(
    const unsigned char* __restrict__ A,       // [M, K_bytes] FP4 packed
    const unsigned char* __restrict__ B,       // [N, K_bytes] FP4 packed
    const unsigned char* __restrict__ A_scale,  // [M, K_groups] raw E8M0
    const unsigned char* __restrict__ B_scale,  // [N, K_groups] raw E8M0
    unsigned short* __restrict__ C,            // [M, N] BF16
    int M, int N, int K_bytes
) {
    // Double-buffered LDS (no padding, use XOR swizzle)
    // A: 64 × 128 = 8192 bytes per buffer
    // B: 128 × 128 = 16384 bytes per buffer
    // Total: 2 × (8192 + 16384) = 49152 bytes = 48KB (fits in 64KB LDS)
    __shared__ unsigned char lds_a[2][TILE_M * K_STEP];
    __shared__ unsigned char lds_b[2][TILE_N * K_STEP];

    // XCD-aware block assignment (8 XCDs on MI355X)
    int bid = blockIdx.x;
    const int npm = M / TILE_M;
    const int npn = N / TILE_N;
    const int NW = gridDim.x;
    const int NXCD = 8;
    if (NW >= NXCD) {
        int remapped = (bid % NXCD) * ((NW + NXCD - 1) / NXCD) + (bid / NXCD);
        if (remapped < NW) bid = remapped;
    }
    const int WGM = (npm < 8) ? npm : 8;
    const int nwig = WGM * npn;
    const int gid = bid / nwig;
    const int fpm = gid * WGM;
    const int gsm = min(npm - fpm, WGM);
    const int pm = fpm + ((bid % nwig) % gsm);
    const int pn = (bid % nwig) / gsm;

    const int tid = threadIdx.x;
    const int wid = tid / 64;
    const int lane = tid % 64;
    const int lane32 = lane & 31;
    const int group = lane >> 5;

    // Warp assignment: 2×4 grid of 32×32 sub-tiles
    const int warp_m = wid / WARPS_N;   // 0 or 1
    const int warp_n = wid % WARPS_N;   // 0-3

    // Output positions for this warp
    const int a_row_base = pm * TILE_M + warp_m * SUB_M;
    const int b_col_base = pn * TILE_N + warp_n * SUB_N;

    const int K_iters = K_bytes / K_STEP;
    const int K_groups = K_bytes / 16;

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float v16f;
    v16f c_acc = {};

    // === Helper: cooperative load with swizzle ===
    // A tile: 64×128 = 8192 bytes. 512 threads × 16B = 8192 → exact 1 round.
    // B tile: 128×128 = 16384 bytes. 512 threads × 16B = 8192 → 2 rounds.

    #define LOAD_A_TILE(buf, k_off) do { \
        int _af = tid * 16; \
        int _ar = _af / K_STEP; \
        int _ac = _af % K_STEP; \
        int _agr = pm * TILE_M + _ar; \
        int _sw = lds_swizzle(_ar, _ac); \
        if (_agr < M) \
            *(uint4*)&lds_a[buf][_sw] = *(const uint4*)&A[_agr * K_bytes + (k_off) + _ac]; \
        else \
            *(uint4*)&lds_a[buf][_sw] = make_uint4(0,0,0,0); \
    } while(0)

    #define LOAD_B_TILE(buf, k_off) do { \
        /* Round 1: first 64 rows */ \
        { int _bf = tid * 16; \
          int _br = _bf / K_STEP; \
          int _bc = _bf % K_STEP; \
          int _bgc = pn * TILE_N + _br; \
          int _sw = lds_swizzle(_br, _bc); \
          if (_bgc < N) \
              *(uint4*)&lds_b[buf][_sw] = *(const uint4*)&B[_bgc * K_bytes + (k_off) + _bc]; \
          else \
              *(uint4*)&lds_b[buf][_sw] = make_uint4(0,0,0,0); \
        } \
        /* Round 2: next 64 rows (offset by 8192 bytes = 64 rows × 128) */ \
        { int _bf = tid * 16 + 8192; \
          int _br = _bf / K_STEP; \
          int _bc = _bf % K_STEP; \
          int _bgc = pn * TILE_N + _br; \
          int _sw = lds_swizzle(_br, _bc); \
          if (_bgc < N) \
              *(uint4*)&lds_b[buf][_sw] = *(const uint4*)&B[_bgc * K_bytes + (k_off) + _bc]; \
          else \
              *(uint4*)&lds_b[buf][_sw] = make_uint4(0,0,0,0); \
        } \
    } while(0)

    // === Load first K-tile ===
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);

    // Conditional stagger: group 1 (warp_m==1) starts half a phase behind
    // This creates the ping-pong: on each SIMD, one wave computes while the other loads
    if (warp_m == 1) {
        __builtin_amdgcn_s_barrier();
    }
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_s_barrier();

    int tic = 0, toc = 1;

    for (int ki = 0; ki < K_iters; ki++, tic ^= 1, toc ^= 1) {
        // --- Memory cluster: prefetch next K-tile ---
        if (ki + 1 < K_iters) {
            int k_next = (ki + 1) * K_STEP;
            LOAD_A_TILE(toc, k_next);
            LOAD_B_TILE(toc, k_next);
        }

        // --- Preload scales for this K-tile into registers ---
        int scale_base = ki * (K_STEP / 16);  // 8 scale groups per K_STEP
        int my_a_row = a_row_base + lane32;
        int my_b_col = b_col_base + lane32;
        unsigned char a_scales[8], b_scales[8];
        if (my_a_row < M) {
            int a_sc_base = my_a_row * K_groups + scale_base;
            #pragma unroll
            for (int s = 0; s < 8; s++)
                a_scales[s] = A_scale[a_sc_base + s];
        } else {
            #pragma unroll
            for (int s = 0; s < 8; s++)
                a_scales[s] = 0;
        }
        if (my_b_col < N) {
            int b_sc_base = my_b_col * K_groups + scale_base;
            #pragma unroll
            for (int s = 0; s < 8; s++)
                b_scales[s] = B_scale[b_sc_base + s];
        } else {
            #pragma unroll
            for (int s = 0; s < 8; s++)
                b_scales[s] = 0;
        }

        // --- Wait for LDS to be ready ---
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // --- Compute cluster: 4 FP4 MFMA sub-iterations ---
        __builtin_amdgcn_s_setprio(1);  // boost compute priority
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int k_byte_start = sub * 32 + group * 16;

            // Read A and B from swizzled LDS
            int a_lds_off = lds_swizzle(warp_m * SUB_M + lane32, k_byte_start);
            int b_lds_off = lds_swizzle(warp_n * SUB_N + lane32, k_byte_start);

            v8i a_reg = {}, b_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_a[tic][a_lds_off];
            *(uint4*)&b_reg = *(uint4*)&lds_b[tic][b_lds_off];

            // Pack scales: broadcast own scale to both bytes (v3 packing)
            unsigned char a_sc = a_scales[sub * 2 + group];
            unsigned char b_sc = b_scales[sub * 2 + group];
            unsigned int pas = (unsigned int)a_sc | ((unsigned int)a_sc << 8);
            unsigned int pbs = (unsigned int)b_sc | ((unsigned int)b_sc << 8);

            __builtin_amdgcn_sched_barrier(0);
            c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, c_acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);  // restore priority
        __builtin_amdgcn_s_barrier();
    }

    // === Store output as BF16 ===
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
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

// === Fused quant kernel (same as proven version) ===
extern "C" __global__ void fused_quant_raw(
    const unsigned short* __restrict__ A, unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As_raw,
    int M, int K, int Kh, int Kg, int Mp) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= Mp * Kg) return;
    int m = gid / Kg; int g = gid % Kg;
    float vals[32]; float amax = 0.0f;
    if (m < M) {
        const uint4* r4 = (const uint4*)(A + m * K + g * 32);
        #pragma unroll
        for (int v=0;v<4;v++){uint4 c=r4[v];unsigned int w[4]={c.x,c.y,c.z,c.w};
            #pragma unroll
            for(int j=0;j<4;j++){float lo=__uint_as_float((w[j]&0xFFFFu)<<16);float hi=__uint_as_float(w[j]&0xFFFF0000u);
                vals[v*8+j*2]=lo;vals[v*8+j*2+1]=hi;amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi)));}}
    } else { for(int i=0;i<32;i++)vals[i]=0.0f; }
    unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
    int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
    As_raw[m*Kg+g]=(unsigned char)sb;
    int qe=127-su;if(qe<1)qe=0;if(qe>254)qe=254;
    float qs=__uint_as_float((unsigned int)qe<<23);if(amax==0.0f)qs=0.0f;
    const unsigned int dmi=149u<<23;float dmf=__uint_as_float(dmi);
    const int vta=((int)(1-127)<<23)+(1<<21)-1;
    auto fp4=[&](float v)->unsigned char{float qf=v*qs;unsigned int qx=__float_as_uint(qf);
        unsigned int s2=qx&0x80000000u;qx^=s2;float qp=__uint_as_float(qx);unsigned char r;
        if(qp>=6.0f)r=0x7;else if(qp<1.0f)r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
        else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
        return(r&0x7)|((unsigned char)(s2>>28)&0x8);};
    unsigned char pk[16];
    #pragma unroll
    for(int i=0;i<16;i++)pk[i]=fp4(vals[2*i])|(fp4(vals[2*i+1])<<4);
    uint4* o4=(uint4*)(Aq+m*Kh+g*16);*o4=*((uint4*)pk);
}

// === PyTorch wrappers ===
static torch::Tensor g_aq, g_asr, g_out8w;
static int g_Mp8=0, g_K8=0, g_N8=0;

torch::Tensor quant_and_8wave_gemm(
    torch::Tensor A,      // [M, K] bf16
    torch::Tensor Bq,     // [N, K/2] uint8 raw FP4
    torch::Tensor Bs,     // [N, K/32] uint8 raw E8M0
    int64_t N_val
) {
    int M_val = (int)A.size(0), K_val = (int)A.size(1);
    int Mp = ((M_val + 63) / 64) * 64;  // pad to TILE_M=64
    int Kh = K_val / 2, Kg = K_val / 32;

    if (Mp != g_Mp8 || K_val != g_K8) {
        g_aq = torch::empty({Mp, Kh}, torch::dtype(torch::kUInt8).device(A.device()));
        g_asr = torch::empty({Mp, Kg}, torch::dtype(torch::kUInt8).device(A.device()));
        g_Mp8 = Mp; g_K8 = K_val;
    }
    if ((int)N_val != g_N8 || Mp != g_Mp8) {
        g_out8w = torch::empty({Mp, (int)N_val}, torch::dtype(torch::kBFloat16).device(A.device()));
        g_N8 = (int)N_val;
    }

    // 1) Launch quant kernel
    int tg = Mp * Kg;
    int th = 64;
    int qbl = (tg + th - 1) / th;
    fused_quant_raw<<<qbl, th, 0, 0>>>(
        (const unsigned short*)A.data_ptr(),
        g_aq.data_ptr<unsigned char>(),
        g_asr.data_ptr<unsigned char>(),
        M_val, K_val, Kh, Kg, Mp);

    // 2) Launch 8-wave GEMM kernel
    int total_tiles = (Mp / TILE_M) * ((int)N_val / TILE_N);
    fp4_8wave<<<total_tiles, NTHREADS, 0, 0>>>(
        g_aq.data_ptr<unsigned char>(),
        (const unsigned char*)Bq.data_ptr(),
        g_asr.data_ptr<unsigned char>(),
        (const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out8w.data_ptr(),
        Mp, (int)N_val, Kh);

    return g_out8w.slice(0, 0, M_val);
}

// GEMM-only entry point (for benchmarking with pre-quantized data)
torch::Tensor run_8wave_gemm(
    torch::Tensor Aq,     // [M, K/2] uint8 FP4
    torch::Tensor Bq,     // [N, K/2] uint8 FP4
    torch::Tensor As,     // [M, K/32] uint8 E8M0
    torch::Tensor Bs,     // [N, K/32] uint8 E8M0
    int64_t N_val
) {
    int M_val = (int)Aq.size(0), Kh = (int)Aq.size(1);
    if (M_val != g_Mp8 || (int)N_val != g_N8) {
        g_out8w = torch::empty({M_val, (int)N_val}, torch::dtype(torch::kBFloat16).device(Aq.device()));
        g_Mp8 = M_val; g_N8 = (int)N_val;
    }
    int total_tiles = (M_val / TILE_M) * ((int)N_val / TILE_N);
    fp4_8wave<<<total_tiles, NTHREADS, 0, 0>>>(
        (const unsigned char*)Aq.data_ptr(),
        (const unsigned char*)Bq.data_ptr(),
        (const unsigned char*)As.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out8w.data_ptr(),
        M_val, (int)N_val, Kh);
    return g_out8w;
}

TORCH_LIBRARY(fp4_8w, m) {
    m.def("quant_gemm", &quant_and_8wave_gemm);
    m.def("gemm_only", &run_8wave_gemm);
}
