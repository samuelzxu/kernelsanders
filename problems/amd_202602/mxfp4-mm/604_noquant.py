#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""#604: No-quant baseline — isolate B load + MFMA time without A quant overhead."""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_nq_src = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#define TM 32
#define TN 32
#define K_STEP 128
#define NW 4
#define NTH (64*NW)
__device__ __forceinline__ int swz_b(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * K_STEP + (col ^ m);
}
extern "C" __global__ __launch_bounds__(NTH, 2)
void noquant(const unsigned char* __restrict__ Bw, unsigned short* __restrict__ C,
    int M, int N, int Kh, int bw_stride) {
    int bid=blockIdx.x;
    const int npm=(M+TM-1)/TM, npn=(N+TN-1)/TN, NWg=gridDim.x;
    if(NWg>=8){int r=(bid%8)*((NWg+7)/8)+(bid/8);if(r<NWg)bid=r;}
    int WGM=min(npm,8),nwig=WGM*npn,gid=bid/nwig,fpm=gid*WGM,gsm=min(npm-fpm,WGM);
    int pm=fpm+((bid%nwig)%gsm), pn=(bid%nwig)/gsm;
    int tid=threadIdx.x, lane=tid%64, l32=lane&31, grp=lane>>5;
    int b_col=pn*TN;
    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc={};
    __shared__ unsigned char lds_b[TN*K_STEP];
    int K_steps=Kh/K_STEP;
    for (int ki=0;ki<K_steps;ki++){
        {
            int flat=tid*16, br=flat/K_STEP, bc=flat%K_STEP;
            int nc=b_col+br, kb=ki*K_STEP+bc;
            if(nc<N&&kb<Kh){
                int sr=nc/16,nw=nc%16,kbb=kb/32,khh=(kb%32)/16;
                const unsigned char*src=Bw+(long)sr*bw_stride+kbb*512+khh*256+nw*16;
                *(uint4*)&lds_b[swz_b(br,bc)]=*(const uint4*)src;
            } else *(uint4*)&lds_b[swz_b(br,bc)]=make_uint4(0,0,0,0);
        }
        __syncthreads();
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for(int sub=0;sub<4;sub++){
            int ko=sub*32+grp*16;
            v8i a_reg={}, b_reg={};  // A is zeros (no quant)
            *(uint4*)&b_reg=*(uint4*)&lds_b[swz_b(l32,ko)];
            __builtin_amdgcn_sched_barrier(0);
            acc=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg,b_reg,acc,4,4,0,127,0,127);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            int r=pm*TM+grp*4+i*8+j, c=b_col+l32;
            if(r<M&&c<N){float val=acc[i*4+j];unsigned int u=__float_as_uint(val);
                u=u+(((u>>16)&1)+0x7FFF);C[r*N+c]=(unsigned short)(u>>16);}
        }
    }
}

static torch::Tensor g_out; static int g_m=0,g_n=0;
torch::Tensor run_noquant(torch::Tensor Bw, torch::Tensor C_dummy, int M, int N, int K) {
    int Kh=K/2;
    if(M!=g_m||N!=g_n){g_out=torch::empty({M,N},torch::dtype(torch::kBFloat16).device(Bw.device()));g_m=M;g_n=N;}
    int grid=((M+32-1)/32)*((N+32-1)/32);
    hipLaunchKernelGGL(noquant,dim3(grid),dim3(256),0,0,
        (const unsigned char*)Bw.data_ptr(),(unsigned short*)g_out.data_ptr(),M,N,Kh,Kh*16);
    return g_out;
}

"""
_nq_cpp = r"""
#include <torch/extension.h>
torch::Tensor run_noquant(torch::Tensor Bw, torch::Tensor C, int M, int N, int K);
"""
_nq = None
try:
    _nq = load_inline(name='nq604',cpp_sources=_nq_cpp,cuda_sources=_nq_src,
        functions=['run_noquant'],verbose=False,
        extra_cuda_cflags=['-O3','-w','-mcumode','--offload-arch=gfx950'])
    print("NoQuant OK")
except Exception as e:
    print(f"NoQuant FAILED: {e}")
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
    if _nq is not None and m == 32 and k == 512:
        return _nq.run_noquant(_ps_cw, A, m, n, k)  # A unused, just for device
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
