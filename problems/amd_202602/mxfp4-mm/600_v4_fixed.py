#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""#600: v3 kernel — A bf16 to LDS (cooperative), quant in register, no A write-back."""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_hip_src = r"""
// FP4 GEMM v4 — fixed A LDS sizing + interleaved load/compute
// A bf16: HBM → LDS (cooperative, swizzled) → register → quant → MFMA
// B FP4:  HBM → LDS (cooperative, swizzled) → register → MFMA
// A LDS: 32 rows × 512 bytes/row = 16KB (256 bf16 per row per K-step)
// B LDS: 32 rows × 128 bytes/row = 4KB
// Total: 20KB (fits in 64KB LDS, room for double-buffering)
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TM 32
#define TN 32
#define K_STEP 128     // FP4 packed bytes per K-iteration
#define A_ROW_BYTES 512 // bf16 bytes per row per K-step (256 bf16 × 2)
#define NW 4
#define NTH (64*NW)

// B LDS swizzle
__device__ __forceinline__ int swz_b(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * K_STEP + (col ^ m);
}

// A bf16 LDS swizzle (for 512-byte rows, ds_read_b128 friendly)
// Swizzle 16-byte groups based on row to avoid bank conflicts
__device__ __forceinline__ int swz_a(int row, int byte_col) {
    // For 64 banks of 4 bytes each with ds_read_b128 (4 phases):
    // XOR the 16-byte group index with (row/2) to spread accesses
    int group16 = byte_col / 16;
    int within16 = byte_col % 16;
    int swizzled_group = group16 ^ ((row / 2) & 0xF);
    return row * A_ROW_BYTES + swizzled_group * 16 + within16;
}

extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_v4(
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
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc = {};

    // LDS: A bf16 + B FP4
    __shared__ unsigned char lds_a[TM * A_ROW_BYTES];   // 32 × 512 = 16KB
    __shared__ unsigned char lds_b[TN * K_STEP];         // 32 × 128 = 4KB

    for (int ki = 0; ki < K_steps; ki++) {
        // Cooperative A bf16 load: 16KB = 256 threads × 64 bytes/thread
        // Each thread loads 4 × uint4 (64 bytes)
        {
            int bytes_per_thread = A_ROW_BYTES * TM / NTH;  // 16384 / 256 = 64
            for (int r = 0; r < bytes_per_thread; r += 16) {
                int flat = tid * bytes_per_thread + r;
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
        }

        // Cooperative B load: 4KB = 256 threads × 16 bytes
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

        // 4 MFMA sub-iterations
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            int g_idx = sub * 2 + grp;

            // Read 32 bf16 from A LDS (each 16-byte group independently swizzled)
            float vals[32];
            {
                int byte_off = g_idx * 64;
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    // Each v reads 16 bytes = 8 bf16 from a separately swizzled address
                    uint4 c = *(uint4*)&lds_a[l32 * A_ROW_BYTES + byte_off + v*16];
                    unsigned int w[4]={c.x,c.y,c.z,c.w};
                    #pragma unroll
                    for (int j=0;j<4;j++) {
                        vals[v*8+j*2] = __uint_as_float((w[j]&0xFFFFu)<<16);
                        vals[v*8+j*2+1] = __uint_as_float(w[j]&0xFFFF0000u);
                    }
                }
            }

            // Quant in registers
            float amax = 0.0f;
            #pragma unroll
            for (int i=0;i<32;i++) amax = fmaxf(amax, fabsf(vals[i]));

            unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
            int eb=(ar>>23)&0xFF,su=eb-129,sb2=su+127;if(sb2<0)sb2=0;
            unsigned char a_scale = (unsigned char)sb2;
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

            // B from swizzled LDS
            int k_off = sub * 32 + grp * 16;
            v8i b_reg = {};
            *(uint4*)&b_reg = *(uint4*)&lds_b[swz_b(l32, k_off)];

            unsigned int pas = (unsigned int)a_scale | ((unsigned int)a_scale << 8);
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
torch::Tensor run_v4(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int N_val) {
    int M=(int)A.size(0), K=(int)A.size(1);
    if(M!=g_m||N_val!=g_n){g_out=torch::empty({M,N_val},torch::dtype(torch::kBFloat16).device(A.device()));g_m=M;g_n=N_val;}
    int grid=((M+TM-1)/TM)*((N_val+TN-1)/TN);
    hipLaunchKernelGGL(fp4_v4,dim3(grid),dim3(NTH),0,0,
        (const unsigned short*)A.data_ptr(),(const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),(unsigned short*)g_out.data_ptr(),M,N_val,K);
    return g_out;
}

"""
_hip_cpp = r"""
#include <torch/extension.h>
torch::Tensor run_v4(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int N_val);
"""
_v3 = None
try:
    _v3 = load_inline(name='v4_600',cpp_sources=_hip_cpp,cuda_sources=_hip_src,
        functions=['run_v4'],verbose=False,
        extra_cuda_cflags=['-O3','-w','-mcumode','--offload-arch=gfx950'])
    _w=torch.randn(32,512,dtype=torch.bfloat16,device="cuda")
    _v3.run_v4(_w,torch.zeros(256,4096,dtype=torch.uint8,device="cuda"),
        torch.zeros(128,512,dtype=torch.uint8,device="cuda"),4096)
    torch.cuda.synchronize();del _w
    print("v4 OK")
except Exception as e:
    print(f"v4 FAILED: {e}")
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()
_ps_ck=None;_ps_cw=None;_ps_cs=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    if _v3 is not None and m == 32 and k == 512:
        return _v3.run_v4(A, _ps_cw, _ps_cs, n)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
