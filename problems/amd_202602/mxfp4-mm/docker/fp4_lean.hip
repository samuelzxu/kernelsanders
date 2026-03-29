// Lean FP4 preshuffle GEMM kernel for MI355X
// Target: shape 3 (M=32, N=4096, K=512)
// Tile: 32x32, K_STEP=256 bytes (512 FP4 = full K in one pass for K=512)
// 4 warps, occupancy 2, interleaved MFMA + memory
//
// Key optimizations from HipKittens 4-wave pattern:
// 1. sched_barrier(0) around every MFMA
// 2. buffer_load_lds for global→LDS (zero register pressure)
// 3. s_setprio(1) during compute
// 4. Precise s_waitcnt with non-zero counts for overlap
// 5. XOR swizzle on LDS for bank-conflict-free ds_read_b128
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TM 32     // Tile M
#define TN 32     // Tile N (one sub-tile per warp, but all warps compute same tile)
#define NWARPS 4
#define NTH (64 * NWARPS)  // 256

// LDS swizzle (from AMD CDNA4 FP8 blog)
__device__ __forceinline__ int swz(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * 128 + (col ^ m);  // row stride = 128 bytes (max K_STEP per 16-row subtile)
}

// buffer_load_lds intrinsic
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_u32 = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc, as3_u32 lds_ptr, int size,
    int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

struct buf_res { uint64_t ptr; uint32_t range; uint32_t config; };
__device__ __forceinline__ i32x4 mk_srsrc(const void* p, uint32_t sz) {
    buf_res r = {(uint64_t)p, sz, 0x110000};
    return *(const i32x4*)&r;
}

// Inline FP4 quant for 32 bf16 values → 16 packed bytes + 1 E8M0 scale
__device__ void quant32(const unsigned short* src, unsigned char* dst, unsigned char& sc) {
    float vals[32]; float amax = 0.0f;
    const uint4* r4 = (const uint4*)src;
    #pragma unroll
    for (int v=0;v<4;v++){uint4 c=r4[v];unsigned int w[4]={c.x,c.y,c.z,c.w};
        #pragma unroll
        for(int j=0;j<4;j++){float lo=__uint_as_float((w[j]&0xFFFFu)<<16);float hi=__uint_as_float(w[j]&0xFFFF0000u);
            vals[v*8+j*2]=lo;vals[v*8+j*2+1]=hi;amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi)));}}
    unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
    int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
    sc=(unsigned char)sb;
    int qe=127-su;if(qe<1)qe=0;if(qe>254)qe=254;
    float qs=__uint_as_float((unsigned int)qe<<23);if(amax==0.0f)qs=0.0f;
    const unsigned int dmi=149u<<23;float dmf=__uint_as_float(dmi);
    const int vta=((int)(1-127)<<23)+(1<<21)-1;
    auto fp4=[&](float v)->unsigned char{float qf=v*qs;unsigned int qx=__float_as_uint(qf);
        unsigned int s2=qx&0x80000000u;qx^=s2;float qp=__uint_as_float(qx);unsigned char r;
        if(qp>=6.0f)r=0x7;else if(qp<1.0f)r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
        else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
        return(r&0x7)|((unsigned char)(s2>>28)&0x8);};
    #pragma unroll
    for(int i=0;i<16;i++)dst[i]=fp4(vals[2*i])|(fp4(vals[2*i+1])<<4);
}

// Main kernel: processes one 32x32 output tile per block
extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_lean_gemm(
    const unsigned short* __restrict__ A,      // [M, K] bf16
    const unsigned char*  __restrict__ Bw,     // [N/16, Kh*16] preshuffle FP4
    const unsigned char*  __restrict__ Bs,     // [N/32, K] shuffled E8M0
    unsigned short*       __restrict__ C,      // [M, N] bf16
    int M, int N, int K
) {
    const int Kh = K / 2;
    const int Kg = K / 32;
    const int bw_stride = Kh * 16;

    // Block assignment with XCD + L2 swizzle
    int bid = blockIdx.x;
    const int npm = (M + TM - 1) / TM, npn = (N + TN - 1) / TN;
    const int NW = gridDim.x, NXCD = 8;
    if (NW >= NXCD) { int r = (bid%NXCD)*((NW+NXCD-1)/NXCD)+(bid/NXCD); if(r<NW) bid=r; }
    int WGM = min(npm, 8), nwig = WGM * npn;
    int gid=bid/nwig, fpm=gid*WGM, gsm=min(npm-fpm,WGM);
    int pm=fpm+((bid%nwig)%gsm), pn=(bid%nwig)/gsm;

    int tid=threadIdx.x, lane=tid%64, l32=lane&31, grp=lane>>5;

    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc = {};

    int a_row_base = pm * TM;
    int b_col_base = pn * TN;

    // For K=512: Kh=256, we process in 2 K-steps of 128 bytes each
    // Each K-step: 4 MFMA sub-iterations (each processes 32 bytes = 64 FP4 values)
    int K_steps = Kh / 128;  // 256/128 = 2 for K=512

    // LDS layout:
    // sa[32 * 384] = A quant data (max Kh=3584 for K=7168, but we cap at 384 for K<=768)
    // sa_scales[32 * 24] = A scales (max Kg=224 for K=7168, cap at 24 for K<=768)
    // sb[32 * 128] = B tile per K-step
    // Total: ~13KB + 768B + 4KB = ~18KB per buffer
    __shared__ unsigned char sa[TM * 384];      // A quant data (Kh up to 384)
    __shared__ unsigned char sa_scales[TM * 24]; // A scales (Kg up to 24)
    __shared__ unsigned char sb[TN * 128];       // B tile per K-step

    {
        int total_groups = TM * Kg;
        for (int base = 0; base < total_groups; base += NTH) {
            int gidx = base + tid;
            if (gidx < total_groups) {
                int row = gidx / Kg;
                int g = gidx % Kg;
                int abs_row = a_row_base + row;
                unsigned char fp4[16]; unsigned char sc;
                if (abs_row < M) {
                    quant32(A + abs_row * K + g * 32, fp4, sc);
                } else {
                    for (int i=0;i<16;i++) fp4[i]=0; sc=0;
                }
                // Store to shared memory (flat, no swizzle needed for A since we index per-lane)
                for (int i=0;i<16;i++) sa[row * 384 + g * 16 + i] = fp4[i];
                sa_scales[row * 24 + g] = sc;
            }
        }
    }
    __syncthreads();

    // Phase 2: Load B tiles and compute
    for (int ki = 0; ki < K_steps; ki++) {
        // Load B tile: 32 rows × 128 bytes = 4096 bytes
        // 256 threads × 16 bytes = 4096 → exact fit
        {
            int flat = tid * 16;
            int br = flat / 128;  // row within tile [0, 32)
            int bc = flat % 128;  // byte within row [0, 128)
            int nc = b_col_base + br;
            int kb = ki * 128 + bc;  // absolute k-byte

            if (nc < N && kb < Kh) {
                // Preshuffle address formula
                int sr = nc / 16, nw = nc % 16;
                // For 16-byte aligned loads: bc is multiple of 16
                int kbb = kb / 32, khh = (kb % 32) / 16;
                const unsigned char* src = Bw + (long)sr * bw_stride + kbb*512 + khh*256 + nw*16;
                *(uint4*)&sb[swz(br, bc)] = *(const uint4*)src;
            } else {
                *(uint4*)&sb[swz(br, bc)] = make_uint4(0,0,0,0);
            }
        }
        __syncthreads();

        // Load scales for this K-step
        int scale_base = ki * 8; // 8 scale groups per K-step of 128 bytes
        unsigned char a_sc[8], b_sc[8];
        {
            int ar = l32;
            if (a_row_base + ar < M) {
                #pragma unroll
                for (int s=0;s<8;s++) a_sc[s] = sa_scales[ar * 24 + scale_base + s];
            } else {
                #pragma unroll
                for (int s=0;s<8;s++) a_sc[s] = 0;
            }
        }
        {
            int nc = b_col_base + l32;
            if (nc < N) {
                int kg8 = Kg / 8;
                int n0=nc/32, n1=(nc&31)/16, n2=nc&15;
                #pragma unroll
                for (int s=0;s<8;s++) {
                    int g = scale_base + s;
                    int g0=g/8, g1=(g&7)/4, g2=g&3;
                    b_sc[s] = Bs[n0*(kg8*256)+g0*256+g2*64+n2*4+g1*2+n1];
                }
            } else {
                #pragma unroll
                for (int s=0;s<8;s++) b_sc[s] = 0;
            }
        }

        // 4 MFMA sub-iterations per K-step
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int ko = sub * 32 + grp * 16;

            // Read A from shared (flat indexing)
            v8i ar = {};
            *(uint4*)&ar = *(uint4*)&sa[l32 * 384 + ki * 128 + ko];

            // Read B from swizzled LDS
            v8i br = {};
            *(uint4*)&br = *(uint4*)&sb[swz(l32, ko)];

            unsigned int pas = (unsigned int)a_sc[sub*2+grp] | ((unsigned int)a_sc[sub*2+grp] << 8);
            unsigned int pbs = (unsigned int)b_sc[sub*2+grp] | ((unsigned int)b_sc[sub*2+grp] << 8);

            __builtin_amdgcn_sched_barrier(0);
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(ar, br, acc, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // Store output as bf16
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = a_row_base + grp * 4 + i * 8 + j;
            int c = b_col_base + l32;
            if (r < M && c < N) {
                float val = acc[i * 4 + j];
                unsigned int u = __float_as_uint(val);
                u = u + (((u >> 16) & 1) + 0x7FFF);
                C[r * N + c] = (unsigned short)(u >> 16);
            }
        }
    }
}
