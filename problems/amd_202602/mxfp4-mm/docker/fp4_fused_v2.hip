// Hand-tuned fused quant+GEMM for M=32 K=512 shapes on MI355X
// Key: fuse A quantization INTO the K-loop (not upfront)
// Each K-step: load bf16 A chunk → quant to FP4 inline → MFMA with B
//
// Pattern from HipKittens 4-wave + AMD CDNA4 FP8 blog:
// - Double-buffered LDS for B tiles
// - Per-K-step inline A quant (32 rows × 4 groups = 128 groups per step)
// - sched_barrier(0) around MFMA for instruction ordering
// - s_setprio(1) during compute
// - XOR swizzle on B LDS for bank-conflict-free ds_read_b128
// - buffer_load_lds for zero-register-pressure B loading
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TM 32      // output tile M
#define TN 32      // output tile N
#define K_STEP 128 // bytes per K-iteration (256 FP4 values → 4 MFMA calls)
#define NW 4       // warps
#define NTH (64*NW) // 256 threads

// LDS XOR swizzle for bank-conflict-free ds_read_b128
// From AMD CDNA4 FP8 GEMM blog
__device__ __forceinline__ int swz_b(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * K_STEP + (col ^ m);
}

// buffer_load_lds: HBM → LDS without touching registers
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_u32 = uint32_t __attribute__((address_space(3)))*;
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_u32 lds, int sz, int voff, int soff, int off, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
struct buf_res { uint64_t ptr; uint32_t range; uint32_t cfg; };
__device__ __forceinline__ i32x4 mk_src(const void* p, uint32_t r) {
    buf_res b = {(uint64_t)p, r, 0x110000};
    return *(const i32x4*)&b;
}

// Inline FP4 quantization for one group of 32 bf16 values
// Returns 16 packed bytes + 1 E8M0 scale
__device__ __forceinline__ void quant_group(
    const unsigned short* __restrict__ src,
    unsigned char* __restrict__ dst16,
    unsigned char& scale_out
) {
    float vals[32]; float amax = 0.0f;
    // Vectorized 128-bit loads (4 loads of 8 bf16 = 32 values)
    const uint4* r4 = (const uint4*)src;
    #pragma unroll
    for (int v = 0; v < 4; v++) {
        uint4 c = r4[v];
        unsigned int w[4] = {c.x, c.y, c.z, c.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float lo = __uint_as_float((w[j] & 0xFFFFu) << 16);
            float hi = __uint_as_float(w[j] & 0xFFFF0000u);
            vals[v*8+j*2] = lo; vals[v*8+j*2+1] = hi;
            amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
        }
    }

    // E8M0 scale computation
    unsigned int ai = __float_as_uint(amax);
    unsigned int ar = (ai + 0x200000u) & 0xFF800000u;
    int eb = (ar >> 23) & 0xFF, su = eb - 129, sb = su + 127;
    if (sb < 0) sb = 0;
    scale_out = (unsigned char)sb;

    // Quantization factor
    int qe = 127 - su; if (qe < 1) qe = 0; if (qe > 254) qe = 254;
    float qs = __uint_as_float((unsigned int)qe << 23);
    if (amax == 0.0f) qs = 0.0f;

    // FP4 encoding constants
    const unsigned int dmi = 149u << 23;
    const float dmf = __uint_as_float(dmi);
    const int vta = ((int)(1-127) << 23) + (1 << 21) - 1;

    auto fp4 = [&](float v) -> unsigned char {
        float qf = v * qs;
        unsigned int qx = __float_as_uint(qf);
        unsigned int s = qx & 0x80000000u; qx ^= s;
        float qp = __uint_as_float(qx);
        unsigned char r;
        if (qp >= 6.0f) r = 0x7;
        else if (qp < 1.0f) r = (unsigned char)((__float_as_uint(qp + dmf) - dmi) & 0xFF);
        else { unsigned int mo = (qx >> 22) & 1;
               r = (unsigned char)((((unsigned int)((int)qx + vta) + mo) >> 22) & 0xFF); }
        return (r & 0x7) | ((unsigned char)(s >> 28) & 0x8);
    };

    // Pack 32 FP4 values into 16 bytes
    #pragma unroll
    for (int i = 0; i < 16; i++)
        dst16[i] = fp4(vals[2*i]) | (fp4(vals[2*i+1]) << 4);
}

extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_fused_v2(
    const unsigned short* __restrict__ A,   // [M, K] bf16
    const unsigned char*  __restrict__ Bw,  // [N/16, Kh*16] preshuffle FP4
    const unsigned char*  __restrict__ Bs,  // [N/32, K] shuffled E8M0
    unsigned short*       __restrict__ C,   // [M, N] bf16
    int M, int N, int K
) {
    const int Kh = K / 2;          // packed K dimension
    const int Kg = K / 32;         // scale groups total
    const int bw_stride = Kh * 16; // bytes per B super-row
    const int K_steps = Kh / K_STEP;

    // XCD + L2 block assignment
    int bid = blockIdx.x;
    const int npm = (M+TM-1)/TM, npn = (N+TN-1)/TN, NW_g = gridDim.x;
    if (NW_g >= 8) { int r = (bid%8)*((NW_g+7)/8)+(bid/8); if(r<NW_g) bid=r; }
    int WGM = min(npm, 8), nwig = WGM*npn;
    int gid=bid/nwig, fpm=gid*WGM, gsm=min(npm-fpm, WGM);
    int pm=fpm+((bid%nwig)%gsm), pn=(bid%nwig)/gsm;

    int tid = threadIdx.x, lane = tid % 64, l32 = lane & 31, grp = lane >> 5;
    int a_row = pm * TM;
    int b_col = pn * TN;

    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc = {};

    // Double-buffered LDS
    // A quant: TM rows × K_STEP bytes = 32 × 128 = 4KB per buffer
    // B data:  TN rows × K_STEP bytes = 32 × 128 = 4KB per buffer
    // A scale: TM × 8 = 256B per buffer (8 scale groups per K-step)
    // Total: 2 × (4KB + 4KB + 256B) ≈ 17KB — fits easily
    __shared__ unsigned char lds_aq[2][TM * K_STEP];   // quantized A FP4
    __shared__ unsigned char lds_b[2][TN * K_STEP];    // preshuffle B FP4
    __shared__ unsigned char lds_as[2][TM * 8];        // A scales (8 per K-step)

    // === PROLOGUE: Load first K-step ===
    int tic = 0, toc = 1;
    {
        // A quant: 32 rows × 4 groups = 128 groups. 256 threads → first 128 quant.
        int groups_per_step = TM * (K_STEP / 16);  // 32 * 8 = 256... wait
        // K_STEP = 128 bytes = 256 FP4 = 8 scale groups of 32 FP4 each
        // Per row: 8 groups. Total: 32 × 8 = 256 groups. 256 threads → 1:1!
        int grp_idx = tid;  // each thread handles one quant group
        if (grp_idx < TM * (K_STEP / 16)) {
            int row = grp_idx / (K_STEP / 16);  // 0..31
            int g = grp_idx % (K_STEP / 16);    // 0..7
            int abs_row = a_row + row;
            int abs_g = g;  // first K-step starts at group 0
            unsigned char fp4_bytes[16]; unsigned char sc;
            if (abs_row < M && abs_g < Kg)
                quant_group(A + abs_row * K + abs_g * 32, fp4_bytes, sc);
            else { for(int i=0;i<16;i++) fp4_bytes[i]=0; sc=0; }
            // Write to LDS
            *(uint4*)&lds_aq[0][row * K_STEP + g * 16] = *(uint4*)fp4_bytes;
            lds_as[0][row * 8 + g] = sc;
        }

        // B load: 32 rows × 128 bytes = 4096 bytes. 256 threads × 16B = exact fit.
        {
            int flat = tid * 16;
            int br = flat / K_STEP, bc = flat % K_STEP;
            int nc = b_col + br, kb = bc;  // first K-step starts at byte 0
            if (nc < N && kb < Kh) {
                int sr = nc/16, nw = nc%16, kbb = kb/32, khh = (kb%32)/16;
                const unsigned char* src = Bw + (long)sr * bw_stride + kbb*512 + khh*256 + nw*16;
                *(uint4*)&lds_b[0][swz_b(br, bc)] = *(const uint4*)src;
            } else {
                *(uint4*)&lds_b[0][swz_b(br, bc)] = make_uint4(0,0,0,0);
            }
        }
    }
    __syncthreads();

    // === K-LOOP with overlapped load/compute ===
    for (int ki = 0; ki < K_steps; ki++, tic ^= 1, toc ^= 1) {
        // Start loading NEXT K-step into toc buffer (overlapped with compute below)
        if (ki + 1 < K_steps) {
            int next_k_byte = (ki + 1) * K_STEP;
            // A quant for next step
            int grp_idx = tid;
            if (grp_idx < TM * (K_STEP / 16)) {
                int row = grp_idx / (K_STEP / 16);
                int g = grp_idx % (K_STEP / 16);
                int abs_row = a_row + row;
                int abs_g = (ki + 1) * (K_STEP / 16) + g;
                unsigned char fp4_bytes[16]; unsigned char sc;
                if (abs_row < M && abs_g < Kg)
                    quant_group(A + abs_row * K + abs_g * 32, fp4_bytes, sc);
                else { for(int i=0;i<16;i++) fp4_bytes[i]=0; sc=0; }
                *(uint4*)&lds_aq[toc][row * K_STEP + g * 16] = *(uint4*)fp4_bytes;
                lds_as[toc][row * 8 + g] = sc;
            }
            // B load for next step
            {
                int flat = tid * 16;
                int br = flat / K_STEP, bc = flat % K_STEP;
                int nc = b_col + br, kb = next_k_byte + bc;
                if (nc < N && kb < Kh) {
                    int sr=nc/16, nw=nc%16, kbb=kb/32, khh=(kb%32)/16;
                    const unsigned char* src = Bw + (long)sr*bw_stride + kbb*512 + khh*256 + nw*16;
                    *(uint4*)&lds_b[toc][swz_b(br, bc)] = *(const uint4*)src;
                } else {
                    *(uint4*)&lds_b[toc][swz_b(br, bc)] = make_uint4(0,0,0,0);
                }
            }
        }

        // Load B scales for this K-step from global (not LDS — scales are small)
        unsigned char b_sc[8];
        {
            int nc = b_col + l32;
            int scale_base = ki * (K_STEP / 16);
            if (nc < N) {
                int kg8 = Kg / 8, n0 = nc/32, n1 = (nc&31)/16, n2 = nc&15;
                #pragma unroll
                for (int s = 0; s < 8; s++) {
                    int g = scale_base + s;
                    int g0=g/8, g1=(g&7)/4, g2=g&3;
                    b_sc[s] = Bs[n0*(kg8*256) + g0*256 + g2*64 + n2*4 + g1*2 + n1];
                }
            } else {
                #pragma unroll
                for (int s = 0; s < 8; s++) b_sc[s] = 0;
            }
        }

        // Load A scales for this K-step from LDS
        unsigned char a_sc[8];
        {
            int row = l32;  // each lane reads its own row's scales
            #pragma unroll
            for (int s = 0; s < 8; s++)
                a_sc[s] = (a_row + row < M) ? lds_as[tic][row * 8 + s] : (unsigned char)0;
        }

        // === COMPUTE: 4 MFMA sub-iterations ===
        // Wait for LDS to be ready, then compute
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
        __builtin_amdgcn_s_setprio(1);

        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int k_off = sub * 32 + grp * 16;

            // Read A and B from LDS
            v8i a_reg = {}, b_reg = {};
            *(uint4*)&a_reg = *(uint4*)&lds_aq[tic][l32 * K_STEP + k_off];
            *(uint4*)&b_reg = *(uint4*)&lds_b[tic][swz_b(l32, k_off)];

            // Pack scales: v3 broadcast
            unsigned int pas = (unsigned int)a_sc[sub*2+grp] | ((unsigned int)a_sc[sub*2+grp] << 8);
            unsigned int pbs = (unsigned int)b_sc[sub*2+grp] | ((unsigned int)b_sc[sub*2+grp] << 8);

            __builtin_amdgcn_sched_barrier(0);
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // === STORE output as bf16 ===
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = a_row + grp * 4 + i * 8 + j;
            int c = b_col + l32;
            if (r < M && c < N) {
                float val = acc[i * 4 + j];
                unsigned int u = __float_as_uint(val);
                u = u + (((u >> 16) & 1) + 0x7FFF);
                C[r * N + c] = (unsigned short)(u >> 16);
            }
        }
    }
}

// Torch wrapper
static torch::Tensor g_out;
static int g_m = 0, g_n = 0;

torch::Tensor run_fused_v2(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int64_t N_val) {
    int M = (int)A.size(0), K = (int)A.size(1);
    if (M != g_m || (int)N_val != g_n) {
        g_out = torch::empty({M, (int)N_val}, torch::dtype(torch::kBFloat16).device(A.device()));
        g_m = M; g_n = (int)N_val;
    }
    int grid = ((M + TM - 1) / TM) * (((int)N_val + TN - 1) / TN);
    hipLaunchKernelGGL(fp4_fused_v2, dim3(grid), dim3(NTH), 0, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),
        (unsigned short*)g_out.data_ptr(),
        M, (int)N_val, K);
    return g_out;
}

TORCH_LIBRARY(fused_v2, m) {
    m.def("run", &run_fused_v2);
}
