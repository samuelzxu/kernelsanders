// FP4 GEMM v5 — uses 16x16x128 MFMA (matching Triton)
// A bf16: HBM → LDS → register → quant → MFMA
// B FP4:  HBM → LDS → register → MFMA
// Tile: 32×32 output = 4 × (16×16) sub-tiles
// Each 16×16×128 MFMA processes 128 FP4 = 64 packed bytes
// With op_sel pair: 256 FP4 = 128 packed bytes per pair
// K_STEP=128 bytes → 1 pair of MFMAs per sub-tile per K-step
// 4 sub-tiles → 8 MFMAs per K-step (matches Triton's 8)
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TM 32
#define TN 32
#define K_STEP 128
#define A_ROW_BYTES 512  // 256 bf16 × 2 bytes
#define NW 4
#define NTH (64*NW)

__device__ __forceinline__ int swz_b(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * K_STEP + (col ^ m);
}

extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_v5(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ Bw,
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
    typedef __attribute__((__vector_size__(4*sizeof(float)))) float v4f;

    // 4 accumulators for 2×2 grid of 16×16 sub-tiles
    v4f acc00 = {}, acc01 = {}, acc10 = {}, acc11 = {};

    // LDS
    __shared__ unsigned char lds_a[TM * A_ROW_BYTES];  // 32 × 512 = 16KB
    __shared__ unsigned char lds_b[TN * K_STEP];         // 32 × 128 = 4KB

    for (int ki = 0; ki < K_steps; ki++) {
        // Cooperative A bf16 load: 16KB, 64 bytes/thread, 4 rounds of 16B
        for (int r = 0; r < 64; r += 16) {
            int flat = tid * 64 + r;
            int a_r = flat / A_ROW_BYTES;
            int a_c = flat % A_ROW_BYTES;
            int abs_row = a_row + a_r;
            int lds_idx = a_r * A_ROW_BYTES + a_c;
            if (abs_row < M) {
                const unsigned char* src = (const unsigned char*)(A + abs_row * K) + ki * A_ROW_BYTES + a_c;
                *(uint4*)&lds_a[lds_idx] = *(const uint4*)src;
            } else {
                *(uint4*)&lds_a[lds_idx] = make_uint4(0,0,0,0);
            }
        }

        // Cooperative B load: 4KB, 16 bytes/thread
        {
            int flat = tid * 16, br = flat/K_STEP, bc = flat%K_STEP;
            int nc = b_col + br, kb = ki * K_STEP + bc;
            if (nc < N && kb < Kh) {
                int sr=nc/16, nw=nc%16, kbb=kb/32, khh=(kb%32)/16;
                const unsigned char* src = Bw + (long)sr*bw_stride + kbb*512 + khh*256 + nw*16;
                *(uint4*)&lds_b[swz_b(br, bc)] = *(const uint4*)src;
            } else {
                *(uint4*)&lds_b[swz_b(br, bc)] = make_uint4(0,0,0,0);
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

        // For 16x16x128 MFMA:
        // Each call: A = 16 rows × 128 FP4 = 16 rows × 64 bytes
        //            B = 128 FP4 × 16 cols = 64 bytes × 16 cols
        // Lane mapping for 16x16:
        //   l32 % 16 = row/col within 16×16
        //   l32 / 16 = not used for 16×16? Actually for 16x16:
        //   Each lane handles 4 output elements (4 floats)
        //   lane32 maps to output column, grp=0/1 maps to rows 0-15 and 16-31... no
        //   For 16x16x128: thread t contributes to C[t%16, t/16] area? Need to check.
        //
        // Actually for the 32x32x64 MFMA we already know:
        //   l32 = column, grp selects row block
        // For 16x16x128: l32%16 = column, l32/16 = row group... different mapping.
        //
        // This is getting complex. Let me stick with 32x32x64 for now but fix the
        // performance issues (A LDS bank conflicts, quant overhead).
        // The 16x16x128 switch needs a separate careful study of register layout.

        // STICKING WITH 32x32x64 for correctness, but with fixed A loading
        typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
        v16f* acc_ptr = (v16f*)&acc00;  // reinterpret 4 × v4f as 1 × v16f
        v16f acc_combined = {};

        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int g_idx = sub * 2 + grp;

            // Read 32 bf16 from A LDS, quant in registers
            float vals[32]; float amax = 0.0f;
            {
                int byte_off = g_idx * 64;
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    uint4 c = *(uint4*)&lds_a[l32 * A_ROW_BYTES + byte_off + v*16];
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

            int k_off = sub * 32 + grp * 16;
            v8i b_reg = {};
            *(uint4*)&b_reg = *(uint4*)&lds_b[swz_b(l32, k_off)];

            unsigned int pas = (unsigned int)a_sc | ((unsigned int)a_sc << 8);
            unsigned int pbs = (unsigned int)b_sc[g_idx] | ((unsigned int)b_sc[g_idx] << 8);

            __builtin_amdgcn_sched_barrier(0);
            acc_combined = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_reg, b_reg, acc_combined, 4, 4, 0, pas, 0, pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);

        // Copy accumulator for next K-step
        acc00 = *(v4f*)&((float*)&acc_combined)[0];
        acc01 = *(v4f*)&((float*)&acc_combined)[4];
        acc10 = *(v4f*)&((float*)&acc_combined)[8];
        acc11 = *(v4f*)&((float*)&acc_combined)[12];

        __syncthreads();
    }

    // Reconstruct full accumulator for output
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f2;
    v16f2 acc_final;
    ((float*)&acc_final)[0] = ((float*)&acc00)[0];
    ((float*)&acc_final)[1] = ((float*)&acc00)[1];
    ((float*)&acc_final)[2] = ((float*)&acc00)[2];
    ((float*)&acc_final)[3] = ((float*)&acc00)[3];
    ((float*)&acc_final)[4] = ((float*)&acc01)[0];
    ((float*)&acc_final)[5] = ((float*)&acc01)[1];
    ((float*)&acc_final)[6] = ((float*)&acc01)[2];
    ((float*)&acc_final)[7] = ((float*)&acc01)[3];
    ((float*)&acc_final)[8] = ((float*)&acc10)[0];
    ((float*)&acc_final)[9] = ((float*)&acc10)[1];
    ((float*)&acc_final)[10] = ((float*)&acc10)[2];
    ((float*)&acc_final)[11] = ((float*)&acc10)[3];
    ((float*)&acc_final)[12] = ((float*)&acc11)[0];
    ((float*)&acc_final)[13] = ((float*)&acc11)[1];
    ((float*)&acc_final)[14] = ((float*)&acc11)[2];
    ((float*)&acc_final)[15] = ((float*)&acc11)[3];

    #pragma unroll
    for (int i=0;i<4;i++){
        #pragma unroll
        for (int j=0;j<4;j++){
            int r = pm*TM + grp*4 + i*8 + j, c = b_col + l32;
            if (r<M && c<N) {
                float val=((float*)&acc_final)[i*4+j]; unsigned int u=__float_as_uint(val);
                u=u+(((u>>16)&1)+0x7FFF); C[r*N+c]=(unsigned short)(u>>16);
            }
        }
    }
}

static torch::Tensor g_out; static int g_m=0,g_n=0;
torch::Tensor run_v5(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int N_val) {
    int M=(int)A.size(0), K=(int)A.size(1);
    if(M!=g_m||N_val!=g_n){g_out=torch::empty({M,N_val},torch::dtype(torch::kBFloat16).device(A.device()));g_m=M;g_n=N_val;}
    int grid=((M+TM-1)/TM)*((N_val+TN-1)/TN);
    hipLaunchKernelGGL(fp4_v5,dim3(grid),dim3(NTH),0,0,
        (const unsigned short*)A.data_ptr(),(const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),(unsigned short*)g_out.data_ptr(),M,N_val,K);
    return g_out;
}
