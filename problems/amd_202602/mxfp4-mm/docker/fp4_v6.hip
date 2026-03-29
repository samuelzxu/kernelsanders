// FP4 GEMM v6 — contiguous B load (no per-byte preshuffle formula)
// Load preshuffle B as raw contiguous bytes → LDS → rearrange during LDS read
// This matches Triton's approach: load contiguous, permute in register
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TM 32
#define TN 32
#define K_STEP 128
#define A_ROW_BYTES 512
#define A_ROW_PAD 528
#define NW 4
#define NTH (64*NW)

__device__ __forceinline__ int swz_b(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * K_STEP + (col ^ m);
}

extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_v6(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ Bw,   // [N/16, Kh*16]
    const unsigned char* __restrict__ Bs,
    unsigned short* __restrict__ C,
    int M, int N, int K
) {
    const int Kh = K/2, Kg = K/32, bw_stride = Kh*16;
    const int K_steps = Kh / K_STEP;

    int bid = blockIdx.x;
    const int npm=(M+TM-1)/TM, npn=(N+TN-1)/TN, NWg=gridDim.x;
    if(NWg>=8){int r=(bid%8)*((NWg+7)/8)+(bid/8);if(r<NWg)bid=r;}
    int WGM=min(npm,8),nwig=WGM*npn,gid=bid/nwig,fpm=gid*WGM,gsm=min(npm-fpm,WGM);
    int pm=fpm+((bid%nwig)%gsm), pn=(bid%nwig)/gsm;

    int tid=threadIdx.x, lane=tid%64, l32=lane&31, grp=lane>>5;
    int a_row = pm*TM, b_col = pn*TN;

    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc = {};

    // LDS: A bf16 (padded) + B raw preshuffle
    __shared__ unsigned char lds_a[TM * A_ROW_PAD];
    // B: load contiguously from preshuffle. TN=32 N-rows = 2 super-rows × K_STEP*16 bytes each
    // Total: 2 * K_STEP * 16 = 2 * 2048 = 4096 bytes
    __shared__ unsigned char lds_b_raw[2 * K_STEP * 16];  // raw preshuffle layout

    for (int ki = 0; ki < K_steps; ki++) {
        // A bf16 cooperative load (coalesced)
        for (int round = 0; round < 4; round++) {
            int flat = (round * NTH + tid) * 16;
            int a_r = flat / A_ROW_BYTES;
            int a_c = flat % A_ROW_BYTES;
            int abs_row = a_row + a_r;
            int lds_idx = a_r * A_ROW_PAD + a_c;
            if (abs_row < M) {
                const unsigned char* src = (const unsigned char*)(A + abs_row * K) + ki * A_ROW_BYTES + a_c;
                *(uint4*)&lds_a[lds_idx] = *(const uint4*)src;
            } else {
                *(uint4*)&lds_a[lds_idx] = make_uint4(0,0,0,0);
            }
        }

        // B preshuffle contiguous load: 4096 bytes = 256 threads × 16 bytes
        {
            int flat = tid * 16;
            int sr_local = flat / (K_STEP * 16);  // 0 or 1 (which super-row)
            int byte_in_sr = flat % (K_STEP * 16); // byte within super-row
            int abs_sr = (b_col / 16) + sr_local;
            // Offset within super-row for this K-step
            int sr_k_offset = ki * K_STEP * 16;
            if (abs_sr < N/16) {
                const unsigned char* src = Bw + abs_sr * bw_stride + sr_k_offset + byte_in_sr;
                *(uint4*)&lds_b_raw[flat] = *(const uint4*)src;
            } else {
                *(uint4*)&lds_b_raw[flat] = make_uint4(0,0,0,0);
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

        // Read B from preshuffle LDS layout
        // In preshuffle: super_row[sr][byte] where byte layout is:
        //   For each 32-byte block (kb): 2 half-blocks (kh=0,1) × 16 N-rows × 16 bytes
        //   Offset: kb*512 + kh*256 + n_within*16 + ki_byte
        // Our l32 maps to n_col within the 32-N-row tile:
        //   n_within = l32 % 16 (position within super-row)
        //   sr_local = l32 / 16 (which super-row: 0 or 1)

        // MFMA sub-iterations
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int g_idx = sub * 2 + grp;

            // Read A bf16 from LDS, quant in registers
            float vals[32]; float amax = 0.0f;
            {
                int byte_off = g_idx * 64;
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    uint4 c = *(uint4*)&lds_a[l32 * A_ROW_PAD + byte_off + v*16];
                    unsigned int w[4]={c.x,c.y,c.z,c.w};
                    #pragma unroll
                    for (int j=0;j<4;j++) {
                        vals[v*8+j*2] = __uint_as_float((w[j]&0xFFFFu)<<16);
                        vals[v*8+j*2+1] = __uint_as_float(w[j]&0xFFFF0000u);
                        amax = fmaxf(amax, fmaxf(fabsf(vals[v*8+j*2]), fabsf(vals[v*8+j*2+1])));
                    }
                }
            }

            // Quant
            unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
            int eb=(ar>>23)&0xFF,su=eb-129,sb2=su+127;if(sb2<0)sb2=0;
            unsigned char a_sc=(unsigned char)sb2;
            int qe=127-su;if(qe<1)qe=0;if(qe>254)qe=254;
            float qs=__uint_as_float((unsigned int)qe<<23);if(amax==0.0f)qs=0.0f;
            const unsigned int dmi=149u<<23;float dmf=__uint_as_float(dmi);
            const int vta=((int)(1-127)<<23)+(1<<21)-1;
            auto fp4=[&](float v)->unsigned char{float qf=v*qs;unsigned int qx=__float_as_uint(qf);
                unsigned int s=qx&0x80000000u;qx^=s;float qp=__uint_as_float(qx);unsigned char r;
                if(qp>=6.0f)r=0x7;else if(qp<1.0f)r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
                else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
                return(r&0x7)|((unsigned char)(s>>28)&0x8);};
            unsigned char packed[16];
            #pragma unroll
            for(int i=0;i<16;i++) packed[i]=fp4(vals[2*i])|(fp4(vals[2*i+1])<<4);
            v8i a_reg = {};
            *(uint4*)&a_reg = *(uint4*)packed;

            // Read B from raw preshuffle LDS
            // l32 = n-column within tile. n_within = l32 % 16, sr_local = l32 / 16
            int n_within = l32 & 15;
            int sr_local = l32 >> 4;
            // For this sub-iteration: k_byte = sub*32 + grp*16 (same as before)
            int k_byte = sub * 32 + grp * 16;
            // Preshuffle offset within the raw LDS:
            int kb = k_byte / 32, kh = (k_byte % 32) / 16;
            int b_lds_off = sr_local * K_STEP * 16 + kb * 512 + kh * 256 + n_within * 16;
            v8i b_reg = {};
            *(uint4*)&b_reg = *(uint4*)&lds_b_raw[b_lds_off];

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
torch::Tensor run_v6(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int N_val) {
    int M=(int)A.size(0), K=(int)A.size(1);
    if(M!=g_m||N_val!=g_n){g_out=torch::empty({M,N_val},torch::dtype(torch::kBFloat16).device(A.device()));g_m=M;g_n=N_val;}
    int grid=((M+TM-1)/TM)*((N_val+TN-1)/TN);
    hipLaunchKernelGGL(fp4_v6,dim3(grid),dim3(NTH),0,0,
        (const unsigned short*)A.data_ptr(),(const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),(unsigned short*)g_out.data_ptr(),M,N_val,K);
    return g_out;
}
