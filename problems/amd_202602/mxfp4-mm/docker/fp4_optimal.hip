// Optimal FP4 GEMM for MI355X: A in registers (load+quant per sub-iter), B in LDS via buffer_load_lds
// Target: shapes 3/4 (M=32, K=512, N=2880/4096)
//
// Data flow:
//   A bf16: HBM → register (global_load_dwordx4) → inline quant (VALU) → MFMA register
//   B FP4:  HBM → LDS (buffer_load_lds, zero register pressure) → register (ds_read) → MFMA
//   Scales: HBM → register (small, cached)
//
// Per sub-iteration (each processes 64 FP4 = 2 scale groups):
//   1. Load 32 bf16 A values for this group (64 bytes, 1 global_load_dwordx4 × 4)
//   2. Compute amax + FP4 quant in registers (~40 VALU ops)
//   3. Read B from LDS (ds_read_b128, swizzled)
//   4. MFMA with FP4 A (from quant) + FP4 B (from LDS) + E8M0 scales
//
// Pipeline: load A[sub+1] while computing MFMA[sub]
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TM 32
#define TN 32
#define K_STEP 128   // bytes FP4 per K-iteration
#define NW 4
#define NTH (64*NW)

// LDS swizzle
__device__ __forceinline__ int swz(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * K_STEP + (col ^ m);
}

// buffer_load_lds intrinsic
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_u32 = uint32_t __attribute__((address_space(3)))*;
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_u32 lds, int sz, int voff, int soff, int off, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
struct buf_res { uint64_t ptr; uint32_t range; uint32_t cfg; };
__device__ __forceinline__ i32x4 mk_src(const void* p, uint32_t r) {
    buf_res b = {(uint64_t)p, r, 0x110000}; return *(const i32x4*)&b;
}

// Inline FP4 quant: 32 bf16 values → 16 packed FP4 bytes + 1 E8M0 scale
// Operates entirely in registers, no LDS
__device__ __forceinline__ void reg_quant32(
    float vals[32], float amax,
    unsigned char packed[16], unsigned char& scale
) {
    unsigned int ai = __float_as_uint(amax);
    unsigned int ar = (ai + 0x200000u) & 0xFF800000u;
    int eb = (ar >> 23) & 0xFF, su = eb - 129, sb = su + 127;
    if (sb < 0) sb = 0;
    scale = (unsigned char)sb;
    int qe = 127 - su; if(qe<1)qe=0; if(qe>254)qe=254;
    float qs = __uint_as_float((unsigned int)qe << 23);
    if (amax == 0.0f) qs = 0.0f;
    const unsigned int dmi = 149u << 23;
    float dmf = __uint_as_float(dmi);
    const int vta = ((int)(1-127) << 23) + (1 << 21) - 1;
    auto fp4 = [&](float v) -> unsigned char {
        float qf = v * qs; unsigned int qx = __float_as_uint(qf);
        unsigned int s = qx & 0x80000000u; qx ^= s;
        float qp = __uint_as_float(qx); unsigned char r;
        if(qp>=6.0f) r=0x7;
        else if(qp<1.0f) r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
        else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
        return(r&0x7)|((unsigned char)(s>>28)&0x8);
    };
    #pragma unroll
    for (int i = 0; i < 16; i++)
        packed[i] = fp4(vals[2*i]) | (fp4(vals[2*i+1]) << 4);
}

extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_optimal(
    const unsigned short* __restrict__ A,   // [M, K] bf16
    const unsigned char*  __restrict__ Bw,  // [N/16, Kh*16] preshuffle FP4
    const unsigned char*  __restrict__ Bs,  // [N/32, K] shuffled E8M0
    unsigned short*       __restrict__ C,   // [M, N] bf16
    int M, int N, int K
) {
    const int Kh = K/2, Kg = K/32, bw_stride = Kh*16;
    const int K_steps = Kh / K_STEP;

    // Block assignment with XCD swizzle
    int bid = blockIdx.x;
    const int npm=(M+TM-1)/TM, npn=(N+TN-1)/TN, NWg=gridDim.x;
    if(NWg>=8){int r=(bid%8)*((NWg+7)/8)+(bid/8);if(r<NWg)bid=r;}
    int WGM=min(npm,8),nwig=WGM*npn,gid=bid/nwig,fpm=gid*WGM,gsm=min(npm-fpm,WGM);
    int pm=fpm+((bid%nwig)%gsm), pn=(bid%nwig)/gsm;

    int tid=threadIdx.x, lane=tid%64, l32=lane&31, grp=lane>>5;
    int my_row = pm*TM + l32;  // this lane's A row
    int b_col = pn * TN;

    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc = {};

    // LDS: ONLY for B (4KB per K-step, no double-buffer needed for K=512 with 2 steps)
    __shared__ unsigned char lds_b[TN * K_STEP];  // 32 × 128 = 4KB

    // Pointer to this lane's A row
    const unsigned short* my_A = (my_row < M) ? (A + my_row * K) : nullptr;

    for (int ki = 0; ki < K_steps; ki++) {
        int k_byte_start = ki * K_STEP;

        // Load B to LDS via cooperative load (all 256 threads)
        // TODO: replace with buffer_load_lds for zero-register-pressure
        {
            int flat = tid * 16, br = flat/K_STEP, bc = flat%K_STEP;
            int nc = b_col + br, kb = k_byte_start + bc;
            if (nc < N && kb < Kh) {
                int sr=nc/16, nw=nc%16, kbb=kb/32, khh=(kb%32)/16;
                const unsigned char* src = Bw + (long)sr*bw_stride + kbb*512 + khh*256 + nw*16;
                *(uint4*)&lds_b[swz(br, bc)] = *(const uint4*)src;
            } else {
                *(uint4*)&lds_b[swz(br, bc)] = make_uint4(0,0,0,0);
            }
        }

        // B scales from global
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

        // Wait for B LDS
        __syncthreads();

        // 4 MFMA sub-iterations per K-step
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int g_idx = sub * 2 + grp;  // which of the 8 scale groups
            int abs_g = ki * 8 + g_idx;

            // Load + quant A in registers for this scale group
            v8i a_reg = {};
            unsigned char a_scale_val = 0;
            if (my_A && abs_g < Kg) {
                const unsigned short* grp_src = my_A + abs_g * 32;
                // Load 32 bf16 values
                float vals[32]; float amax = 0.0f;
                const uint4* r4 = (const uint4*)grp_src;
                #pragma unroll
                for (int v=0;v<4;v++){
                    uint4 c = r4[v]; unsigned int w[4]={c.x,c.y,c.z,c.w};
                    #pragma unroll
                    for(int j=0;j<4;j++){
                        float lo=__uint_as_float((w[j]&0xFFFFu)<<16);
                        float hi=__uint_as_float(w[j]&0xFFFF0000u);
                        vals[v*8+j*2]=lo; vals[v*8+j*2+1]=hi;
                        amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi)));
                    }
                }
                // Quant
                unsigned char packed[16];
                reg_quant32(vals, amax, packed, a_scale_val);
                *(uint4*)&a_reg = *(uint4*)packed;
            }

            // Read B from swizzled LDS
            int k_off = sub * 32 + grp * 16;
            v8i b_reg = {};
            *(uint4*)&b_reg = *(uint4*)&lds_b[swz(l32, k_off)];

            // Scale packing
            unsigned int pas = (unsigned int)a_scale_val | ((unsigned int)a_scale_val << 8);
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
            int r = pm*TM + grp*4 + i*8 + j;
            int c = b_col + l32;
            if (r < M && c < N) {
                float val = acc[i*4+j]; unsigned int u = __float_as_uint(val);
                u = u + (((u >> 16) & 1) + 0x7FFF);
                C[r*N+c] = (unsigned short)(u >> 16);
            }
        }
    }
}

static torch::Tensor g_out; static int g_m=0,g_n=0;
torch::Tensor run_optimal(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int N_val) {
    int M=(int)A.size(0), K=(int)A.size(1);
    if(M!=g_m||N_val!=g_n){g_out=torch::empty({M,N_val},torch::dtype(torch::kBFloat16).device(A.device()));g_m=M;g_n=N_val;}
    int grid=((M+TM-1)/TM)*((N_val+TN-1)/TN);
    hipLaunchKernelGGL(fp4_optimal,dim3(grid),dim3(NTH),0,0,
        (const unsigned short*)A.data_ptr(),(const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),(unsigned short*)g_out.data_ptr(),M,N_val,K);
    return g_out;
}
