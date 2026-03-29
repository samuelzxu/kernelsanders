// FP4 GEMM v3 — matches Triton's actual strategy:
//   A bf16: HBM → LDS (cooperative coalesced) → register → quant (register) → MFMA
//   B FP4:  HBM → LDS (cooperative coalesced) → register → MFMA
//   Key: A only does LDS READ (no write-back of quant). Quant result stays in registers.
//   Uses 16×16×128 MFMA (v_mfma_scale_f32_16x16x128_f8f6f4) — same as Triton
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Tile config for 16x16 MFMA
#define TM 32      // output M per block (2 × 16)
#define TN 32      // output N per block (2 × 16)
#define K_STEP 128 // packed FP4 bytes per iteration = 256 FP4 values
#define NW 4
#define NTH (64*NW)

// LDS swizzle for B
__device__ __forceinline__ int swz(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * K_STEP + (col ^ m);
}

// Inline quant: 32 bf16 → 16 packed FP4 bytes + scale
// Input: 32 bf16 from LDS (already loaded to registers)
__device__ __forceinline__ void quant32_inline(
    float vals[32], unsigned char packed[16], unsigned char& scale
) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(vals[i]));

    unsigned int ai = __float_as_uint(amax);
    unsigned int ar = (ai + 0x200000u) & 0xFF800000u;
    int eb = (ar >> 23) & 0xFF, su = eb - 129, sb = su + 127;
    if (sb < 0) sb = 0;
    scale = (unsigned char)sb;
    int qe = 127 - su; if (qe < 1) qe = 0; if (qe > 254) qe = 254;
    float qs = __uint_as_float((unsigned int)qe << 23);
    if (amax == 0.0f) qs = 0.0f;

    const unsigned int dmi = 149u << 23;
    float dmf = __uint_as_float(dmi);
    const int vta = ((int)(1-127) << 23) + (1 << 21) - 1;
    auto fp4 = [&](float v) -> unsigned char {
        float qf = v * qs; unsigned int qx = __float_as_uint(qf);
        unsigned int s = qx & 0x80000000u; qx ^= s;
        float qp = __uint_as_float(qx); unsigned char r;
        if (qp >= 6.0f) r = 0x7;
        else if (qp < 1.0f) r = (unsigned char)((__float_as_uint(qp + dmf) - dmi) & 0xFF);
        else { unsigned int mo = (qx >> 22) & 1;
               r = (unsigned char)((((unsigned int)((int)qx + vta) + mo) >> 22) & 0xFF); }
        return (r & 0x7) | ((unsigned char)(s >> 28) & 0x8);
    };
    #pragma unroll
    for (int i = 0; i < 16; i++)
        packed[i] = fp4(vals[2*i]) | (fp4(vals[2*i+1]) << 4);
}

extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_v3(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ Bw,
    const unsigned char* __restrict__ Bs,
    unsigned short* __restrict__ C,
    int M, int N, int K
) {
    const int Kh = K/2, Kg = K/32, bw_stride = Kh*16;
    const int K_steps = Kh / K_STEP;
    // A bf16: M×K elements, stride K per row. Each bf16 = 2 bytes.
    // Per K-step: 256 bf16 values per row = 512 bytes per row.
    // For TM=32 rows: 32 × 512 = 16KB of A bf16 per K-step.

    int bid = blockIdx.x;
    const int npm=(M+TM-1)/TM, npn=(N+TN-1)/TN, NWg=gridDim.x;
    if (NWg>=8){int r=(bid%8)*((NWg+7)/8)+(bid/8);if(r<NWg)bid=r;}
    int WGM=min(npm,8),nwig=WGM*npn,gid=bid/nwig,fpm=gid*WGM,gsm=min(npm-fpm,WGM);
    int pm=fpm+((bid%nwig)%gsm), pn=(bid%nwig)/gsm;

    int tid=threadIdx.x, lane=tid%64, l32=lane&31, grp=lane>>5;
    int a_row = pm*TM, b_col = pn*TN;

    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc = {};

    // LDS layout:
    // A bf16: TM × (K_STEP*2) bytes = 32 × 256 = 8192 bytes (bf16 for this K-step)
    // B FP4:  TN × K_STEP bytes = 32 × 128 = 4096 bytes
    // Total per buffer: 12288 bytes = 12KB
    __shared__ unsigned char lds_a_bf16[TM * K_STEP * 2];  // 32 × 256 = 8KB bf16
    __shared__ unsigned char lds_b[TN * K_STEP];             // 32 × 128 = 4KB FP4

    for (int ki = 0; ki < K_steps; ki++) {
        int k_elem_start = ki * K_STEP * 2;  // K_STEP * 2 = 256 bf16 elements

        // Cooperative A bf16 load: 8KB = 256 threads × 32 bytes per thread
        {
            int flat = tid * 32;  // 32 bytes per thread
            int a_r = flat / (K_STEP * 2);  // row within tile
            int a_c = flat % (K_STEP * 2);  // byte offset within row
            int abs_row = a_row + a_r;
            if (abs_row < M) {
                // A is [M, K] bf16. Row stride = K * 2 bytes.
                const unsigned char* src = (const unsigned char*)(A + abs_row * K) + ki * K_STEP * 2 + a_c;
                *(uint4*)&lds_a_bf16[a_r * K_STEP * 2 + a_c] = *(const uint4*)src;
                *(uint4*)&lds_a_bf16[a_r * K_STEP * 2 + a_c + 16] = *(const uint4*)(src + 16);
            } else {
                *(uint4*)&lds_a_bf16[a_r * K_STEP * 2 + a_c] = make_uint4(0,0,0,0);
                *(uint4*)&lds_a_bf16[a_r * K_STEP * 2 + a_c + 16] = make_uint4(0,0,0,0);
            }
        }

        // Cooperative B load: 4KB = 256 threads × 16 bytes
        {
            int flat = tid * 16;
            int br = flat / K_STEP, bc = flat % K_STEP;
            int nc = b_col + br, kb = ki * K_STEP + bc;
            if (nc < N && kb < Kh) {
                int sr=nc/16, nw=nc%16, kbb=kb/32, khh=(kb%32)/16;
                const unsigned char* src = Bw + (long)sr*bw_stride + kbb*512 + khh*256 + nw*16;
                *(uint4*)&lds_b[swz(br, bc)] = *(const uint4*)src;
            } else {
                *(uint4*)&lds_b[swz(br, bc)] = make_uint4(0,0,0,0);
            }
        }

        // B scales
        unsigned char b_sc[8];
        {
            int nc = b_col + l32, sb_off = ki * 8;
            if (nc < N) {
                int kg8=Kg/8, n0=nc/32, n1=(nc&31)/16, n2=nc&15;
                #pragma unroll
                for (int s=0;s<8;s++){
                    int g=sb_off+s, g0=g/8, g1=(g&7)/4, g2=g&3;
                    b_sc[s]=Bs[n0*(kg8*256)+g0*256+g2*64+n2*4+g1*2+n1];
                }
            } else { for(int s=0;s<8;s++) b_sc[s]=0; }
        }

        __syncthreads();

        // 4 MFMA sub-iterations per K-step
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int g_idx = sub * 2 + grp;

            // Read 32 bf16 values from A LDS for this scale group
            // Each lane reads its own row (l32), group g_idx
            float vals[32];
            {
                const unsigned short* a_lds = (const unsigned short*)lds_a_bf16;
                int base = l32 * (K_STEP * 2 / 2) + g_idx * 32;  // 32 bf16 values
                const uint4* r4 = (const uint4*)(a_lds + base);
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    uint4 c = r4[v]; unsigned int w[4] = {c.x, c.y, c.z, c.w};
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        vals[v*8+j*2] = __uint_as_float((w[j] & 0xFFFFu) << 16);
                        vals[v*8+j*2+1] = __uint_as_float(w[j] & 0xFFFF0000u);
                    }
                }
            }

            // Quant in registers
            unsigned char packed[16]; unsigned char a_sc;
            quant32_inline(vals, packed, a_sc);

            // A → MFMA register (16 bytes in lower half of v8i)
            v8i a_reg = {};
            *(uint4*)&a_reg = *(uint4*)packed;

            // B from swizzled LDS
            int k_off = sub * 32 + grp * 16;
            v8i b_reg = {};
            *(uint4*)&b_reg = *(uint4*)&lds_b[swz(l32, k_off)];

            unsigned int pas = (unsigned int)a_sc | ((unsigned int)a_sc << 8);
            unsigned int pbs = (unsigned int)b_sc[g_idx] | ((unsigned int)b_sc[g_idx] << 8);

            __builtin_amdgcn_sched_barrier(0);
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // Store bf16
    #pragma unroll
    for (int i=0;i<4;i++){
        #pragma unroll
        for (int j=0;j<4;j++){
            int r = pm*TM + grp*4 + i*8 + j, c = b_col + l32;
            if (r<M && c<N) {
                float val=acc[i*4+j]; unsigned int u=__float_as_uint(val);
                u=u+(((u>>16)&1)+0x7FFF); C[r*N+c]=(unsigned short)(u>>16);
            }
        }
    }
}

static torch::Tensor g_out; static int g_m=0,g_n=0;
torch::Tensor run_v3(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int N_val) {
    int M=(int)A.size(0), K=(int)A.size(1);
    if(M!=g_m||N_val!=g_n){g_out=torch::empty({M,N_val},torch::dtype(torch::kBFloat16).device(A.device()));g_m=M;g_n=N_val;}
    int grid=((M+TM-1)/TM)*((N_val+TN-1)/TN);
    hipLaunchKernelGGL(fp4_v3,dim3(grid),dim3(NTH),0,0,
        (const unsigned short*)A.data_ptr(),(const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),(unsigned short*)g_out.data_ptr(),M,N_val,K);
    return g_out;
}
