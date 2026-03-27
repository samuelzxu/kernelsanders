#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Debug rawB with v3 scale packing on K=1536 M=256"""
import torch, os, traceback
from task import input_t, output_t
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.shuffle import shuffle_weight
from torch.utils.cpp_extension import load_inline

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

_hip = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

extern "C" __global__ void rawb_v3(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ B_q,
    const unsigned char* __restrict__ B_scale,
    float* __restrict__ C,
    int M, int N, int K
) {
    int tile_n = blockIdx.x * 32;
    int tile_m = blockIdx.y * 32;
    int lane = threadIdx.x;
    int lane32 = lane & 31;
    int group = lane >> 5;
    int K_half = K / 2;
    int K_groups = K / 32;

    float acc[16];
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int ki = 0; ki < K/64; ki++) {
        int k_base = ki * 64;
        int a_row = tile_m + lane32;

        // A quant
        float vals[32]; float amax = 0.0f;
        unsigned char a_fp4[16]; unsigned char a_sc;
        if (a_row < M) {
            const uint4* ap4 = (const uint4*)(A + a_row * K + k_base + group * 32);
            for (int v=0;v<4;v++) { uint4 c=ap4[v]; unsigned int w[4]={c.x,c.y,c.z,c.w};
                for (int j=0;j<4;j++) { float lo=__uint_as_float((w[j]&0xFFFFu)<<16); float hi=__uint_as_float(w[j]&0xFFFF0000u);
                    vals[v*8+j*2]=lo;vals[v*8+j*2+1]=hi;amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi))); }}
        } else { for(int i=0;i<32;i++)vals[i]=0.0f; }
        unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
        int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;a_sc=(unsigned char)sb;
        int qe=127-su;if(qe<1)qe=0;if(qe>254)qe=254;
        float qs=__uint_as_float((unsigned int)qe<<23);if(amax==0.0f)qs=0.0f;
        const unsigned int dmi=149u<<23;float dmf=__uint_as_float(dmi);
        const int vta=((int)(1-127)<<23)+(1<<21)-1;
        for(int i=0;i<16;i++){
            auto fp4=[&](float v)->unsigned char{float qf=v*qs;unsigned int qx=__float_as_uint(qf);
                unsigned int s=qx&0x80000000u;qx^=s;float qp=__uint_as_float(qx);unsigned char r;
                if(qp>=6.0f)r=0x7;else if(qp<1.0f)r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
                else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
                return(r&0x7)|((unsigned char)(s>>28)&0x8);};
            a_fp4[i]=fp4(vals[2*i])|(fp4(vals[2*i+1])<<4);}

        // B raw load
        unsigned char b_fp4[16]; unsigned char b_sc;
        int b_col = tile_n + lane32;
        if(b_col<N){
            const unsigned char* bp=B_q+b_col*K_half+(k_base+group*32)/2;
            for(int i=0;i<16;i++)b_fp4[i]=bp[i];
            b_sc=B_scale[b_col*K_groups+(k_base/32)+group];
        } else { for(int i=0;i<16;i++)b_fp4[i]=0;b_sc=127; }

        // v3 scale packing
        unsigned int pas=(unsigned int)a_sc|((unsigned int)a_sc<<8);
        unsigned int pbs=(unsigned int)b_sc|((unsigned int)b_sc<<8);

        typedef int v8si __attribute__((ext_vector_type(8)));
        typedef float v16f __attribute__((vector_size(64)));
        v8si areg={};areg[0]=*(int*)&a_fp4[0];areg[1]=*(int*)&a_fp4[4];areg[2]=*(int*)&a_fp4[8];areg[3]=*(int*)&a_fp4[12];
        v8si breg={};breg[0]=*(int*)&b_fp4[0];breg[1]=*(int*)&b_fp4[4];breg[2]=*(int*)&b_fp4[8];breg[3]=*(int*)&b_fp4[12];
        v16f creg;for(int i=0;i<16;i++)creg[i]=acc[i];
        creg=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(areg,breg,creg,4,4,0,pas,0,pbs);
        for(int i=0;i<16;i++)acc[i]=creg[i];
    }

    for(int i=0;i<4;i++)for(int j=0;j<4;j++){
        int r=tile_m+group*4+i*8+j;int c=tile_n+lane32;
        if(r<M&&c<N)C[r*N+c]=acc[i*4+j];}
}

torch::Tensor run(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int M, int N, int K) {
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
    dim3 grid((N+31)/32, (M+31)/32, 1);
    rawb_v3<<<grid, 64, 0, 0>>>((const unsigned short*)A.data_ptr(),
        (const unsigned char*)Bq.data_ptr(),(const unsigned char*)Bs.data_ptr(),
        C.data_ptr<float>(), M, N, K);
    return C;
}
"""

_cpp = r"""
#include <torch/extension.h>
torch::Tensor run(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int M, int N, int K);
"""

try:
    mod = load_inline(name='rawbv3_492', cpp_sources=_cpp, cuda_sources=_hip,
        functions=['run'], verbose=False,
        extra_cuda_cflags=['-O3','-w','-mcumode','--offload-arch=gfx950'])
    M,N,K = 256, 3072, 1536
    A = torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
    B = torch.randn(N,K,dtype=torch.bfloat16,device="cuda")
    B_q,B_s = dynamic_mxfp4_quant(B)
    B_q_t = B_q.view(dtypes.fp4x2)
    B_s_sh = e8m0_shuffle(B_s).view(dtypes.fp8_e8m0)
    B_shuf = shuffle_weight(B_q_t, layout=(16,16))
    A_q,A_s = dynamic_mxfp4_quant(A)
    A_q_t = A_q.view(dtypes.fp4x2)
    A_s_sh = e8m0_shuffle(A_s).view(dtypes.fp8_e8m0)
    ref = aiter.gemm_a4w4(A_q_t,B_shuf,A_s_sh,B_s_sh,dtype=dtypes.bf16,bpreshuffle=True)
    # Raw B_scale (unshuffle)
    bsu = B_s.view(torch.uint8)
    # dynamic_mxfp4_quant returns [N_pad, K_groups] - take [:N, :K//32]
    B_s_raw = bsu[:N, :K//32].contiguous()
    custom = mod.run(A, B_q.view(torch.uint8)[:N,:K//2].contiguous(), B_s_raw, M, N, K)
    torch.cuda.synchronize()
    diff = torch.abs(ref.float() - custom.to(torch.bfloat16).float())
    print(f"Max diff: {diff.max().item():.4f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    print(f"Ref sum: {ref.float().sum().item():.1f}")
    print(f"Custom sum: {custom.sum().item():.1f}")
    # Tolerance check
    max_abs = torch.max(torch.abs(ref.float()), torch.abs(custom.to(torch.bfloat16).float()))
    rel_err = diff / (max_abs + 1e-6)
    print(f"Max rel err: {rel_err.max().item():.4f}")
    pct_within = (diff < 0.01 * max_abs + 0.01).float().mean().item() * 100
    print(f"% within rtol=1e-2 atol=1e-2: {pct_within:.1f}%")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
import json
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_ck=None;_cw=None;_cs=None
@torch.inference_mode()
def custom_kernel(data):
    global _ck,_cw,_cs
    A2=data[0];B2=data[3];Bs=data[4]
    m,k=A2.shape;n=data[1].shape[0]
    dp=B2.data_ptr()
    if dp!=_ck:
        _ck=dp;_cw=B2.view(torch.uint8).reshape(n//16,(k//2)*16)
        _cs=Bs.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A2,_cw,_cs,prequant=True,dtype=torch.bfloat16)
