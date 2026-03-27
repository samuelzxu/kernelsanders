// FUSED: bf16 A quant + FP4 GEMM in ONE kernel
// Loads bf16 A from global, quantizes to FP4 inline, stores to LDS
// B is pre-quantized FP4 (loaded directly to LDS)
// Single kernel launch = no quant+GEMM dispatch gap
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TILE_M 32
#define TILE_N 32
#define K_STEP 128   // 128 bytes of FP4 packed = 256 FP4 elements = 1536/6 K iterations
#define NWARPS 4
#define NTHREADS (64 * NWARPS)

extern "C" __global__ __launch_bounds__(NTHREADS, 4)
void fp4_fused_quant_gemm(
    const unsigned short* __restrict__ A_bf16,  // [M, K] bf16
    const unsigned char* __restrict__ B_q,      // [N, K/2] FP4 packed
    const unsigned char* __restrict__ B_scale,  // [N, K/32] E8M0 raw
    unsigned short* __restrict__ C,             // [M, N] bf16
    int M, int N, int K
) {
    __shared__ unsigned char lds_a[2][TILE_M][K_STEP + 16];  // FP4 packed after quant
    __shared__ unsigned char lds_b[2][TILE_N][K_STEP + 16];  // FP4 packed from B
    __shared__ unsigned char lds_as[2][TILE_M][K_STEP/16];   // A scales per tile

    int K_bytes = K / 2;
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
    int KI = K_bytes / K_STEP;
    int Kg = K / 32;

    // === Inline quant helpers ===
    // Quant function: bf16 group of 32 → 16 packed FP4 bytes + 1 E8M0 scale
    // Each thread handles one group (32 bf16 values)
    auto quant_group = [&](const unsigned short* src, unsigned char* dst_fp4, unsigned char* dst_scale) {
        float vals[32]; float amax = 0.0f;
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            uint4 chunk = ((const uint4*)src)[v];
            unsigned int w[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float lo = __uint_as_float((w[j] & 0xFFFFu) << 16);
                float hi = __uint_as_float(w[j] & 0xFFFF0000u);
                vals[v*8+j*2] = lo; vals[v*8+j*2+1] = hi;
                amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
            }
        }
        unsigned int ai = __float_as_uint(amax);
        unsigned int ar = (ai + 0x200000u) & 0xFF800000u;
        int eb = (ar >> 23) & 0xFF, su = eb - 129, sb = su + 127;
        if (sb < 0) sb = 0;
        *dst_scale = (unsigned char)sb;
        int qe = 127 - su; if(qe<1)qe=0; if(qe>254)qe=254;
        float qs = __uint_as_float((unsigned int)qe << 23);
        if (amax == 0.0f) qs = 0.0f;
        const unsigned int dmi = 149u << 23;
        float dmf = __uint_as_float(dmi);
        const int vta = ((int)(1-127) << 23) + (1 << 21) - 1;
        auto fp4 = [&](float v) -> unsigned char {
            float qf = v * qs; unsigned int qx = __float_as_uint(qf);
            unsigned int s2 = qx & 0x80000000u; qx ^= s2;
            float qp = __uint_as_float(qx); unsigned char r;
            if(qp>=6.0f)r=0x7;else if(qp<1.0f)r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
            else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
            return(r&0x7)|((unsigned char)(s2>>28)&0x8);
        };
        #pragma unroll
        for (int i = 0; i < 16; i++)
            dst_fp4[i] = fp4(vals[2*i]) | (fp4(vals[2*i+1]) << 4);
    };

    // === Load + quant first K-tile ===
    {
        // A: load bf16, quant to FP4, store to LDS
        // Each thread handles one group of 32 bf16 = 64 bytes in, 16 bytes out + 1 scale
        // 256 threads can handle 256 groups. Tile has 32*K_STEP/16 = 32*8 = 256 groups. Perfect!
        int group_idx = tid;  // 0..255
        int a_r = group_idx / (K_STEP / 16);  // row 0..31
        int a_g = group_idx % (K_STEP / 16);  // group within row 0..7
        int a_gr = arb + a_r;
        if (a_gr < M) {
            const unsigned short* src = A_bf16 + a_gr * K + a_g * 32;
            quant_group(src, &lds_a[0][a_r][a_g * 16], &lds_as[0][a_r][a_g]);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) lds_a[0][a_r][a_g * 16 + i] = 0;
            lds_as[0][a_r][a_g] = 127;
        }

        // B: load pre-quantized FP4 directly
        int b_flat = tid * 16;
        int b_r = b_flat / K_STEP, b_c = b_flat % K_STEP;
        int b_gc = bcb + b_r;
        *(uint4*)&lds_b[0][b_r][b_c] = (b_gc < N) ?
            *(const uint4*)&B_q[b_gc * K_bytes + b_c] : make_uint4(0,0,0,0);
    }
    __syncthreads();

    int tic = 0, toc = 1;
    for (int ki = 0; ki < KI; ki++, tic ^= 1, toc ^= 1) {
        // Prefetch next K-tile
        if (ki + 1 < KI) {
            int k_next = (ki + 1) * K_STEP;
            int k_next_bf16 = k_next * 2;  // bf16 offset = byte offset * 2

            int group_idx = tid;
            int a_r = group_idx / (K_STEP / 16);
            int a_g = group_idx % (K_STEP / 16);
            int a_gr = arb + a_r;
            if (a_gr < M) {
                const unsigned short* src = A_bf16 + a_gr * K + (ki + 1) * K_STEP * 2 + a_g * 32;
                // Wait — K_STEP is in FP4 packed bytes. For bf16: K_STEP bytes of FP4 = K_STEP*2 bf16 elements
                // Actually: K_STEP=128 packed bytes = 256 FP4 elements. In bf16: 256 elements * 2 bytes = 512 bytes
                // So bf16 offset = ki * 256 (elements) + a_g * 32 (elements per group)
                const unsigned short* src2 = A_bf16 + a_gr * K + (ki + 1) * 256 + a_g * 32;
                quant_group(src2, &lds_a[toc][a_r][a_g * 16], &lds_as[toc][a_r][a_g]);
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) lds_a[toc][a_r][a_g * 16 + i] = 0;
                lds_as[toc][a_r][a_g] = 127;
            }

            int b_flat = tid * 16;
            int b_r = b_flat / K_STEP, b_c = b_flat % K_STEP;
            int b_gc = bcb + b_r;
            *(uint4*)&lds_b[toc][b_r][b_c] = (b_gc < N) ?
                *(const uint4*)&B_q[b_gc * K_bytes + (ki + 1) * K_STEP + b_c] : make_uint4(0,0,0,0);
        }

        // Load A scales from LDS and B scales from global
        int sb2 = ki * (K_STEP / 16);
        unsigned char as8[8], bs8[8];
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            as8[s] = lds_as[tic][lane32][s];  // A scale from LDS (computed during quant)
            bs8[s] = B_scale[(bcb + lane32) * Kg + sb2 + s];
        }

        // Compute
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int kbs = sub * 32 + group * 16;
            v8i a_reg = {}, b_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_a[tic][lane32][kbs];
            *(uint4*)&b_reg = *(uint4*)&lds_b[tic][lane32][kbs];
            unsigned char asc = as8[sub * 2 + group], bsc = bs8[sub * 2 + group];
            unsigned int pas = (unsigned int)asc | ((unsigned int)asc << 8);
            unsigned int pbs = (unsigned int)bsc | ((unsigned int)bsc << 8);
            __builtin_amdgcn_sched_barrier(0);
            c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, c_acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
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

torch::Tensor run_fused(torch::Tensor A_bf16, torch::Tensor B_q, torch::Tensor B_scale, int64_t N_val) {
    int M_val = (int)A_bf16.size(0), K_val = (int)A_bf16.size(1);
    if (M_val != g_m || (int)N_val != g_n) {
        g_out = torch::empty({M_val, N_val}, torch::dtype(torch::kBFloat16).device(A_bf16.device()));
        g_m = M_val; g_n = (int)N_val;
    }
    int total_tiles = (M_val / TILE_M) * ((int)N_val / TILE_N);
    fp4_fused_quant_gemm<<<total_tiles, NTHREADS, 0, 0>>>(
        (const unsigned short*)A_bf16.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_scale.data_ptr(),
        (unsigned short*)g_out.data_ptr(),
        M_val, (int)N_val, K_val);
    return g_out;
}

TORCH_LIBRARY(hk_gemm, m) {
    m.def("run_fused", &run_fused);
}
