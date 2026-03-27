#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#554: Config sweep for shapes 2 and 5.
Tests untried combos: BSN=64 for shape2, stages=3/KSPLIT=3 for shape5.
"""
import torch, os, json, time
from task import input_t, output_t

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)

# === CONFIG VARIANTS TO TEST ===
# Shape 2: M=16 N=2112 K=7168 — current best 13.2us (BSN=128 KS=8 waves=3)
# Shape 5: M=64 N=7168 K=2048 — current best 13.1us (BSN=256 KS=2 mfma16 waves=3)

# Variant A: shape2 BSN=64 (more N-blocks: 33 vs 16)
_cfgA={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},
"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},
"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":3,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},
"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}

# Changes from proven config (#411):
# Shape 2: BSN=128→64 (2x more N-blocks)
# Shape 5: stages=2→3 (deeper pipeline)
# Everything else unchanged

for _sk,_cfg in _cfgA.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warmup all shapes
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()

# Quick self-benchmark to print timing before eval runs
print("=== Quick self-benchmark (variant A) ===")
for _m,_n,_k in [(16,2112,7168),(64,7168,2048)]:
    _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
    _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
    _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
    # warmup
    for _ in range(3):
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    torch.cuda.synchronize()
    # measure
    st=torch.cuda.Event(enable_timing=True)
    en=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    st.record()
    for _ in range(10):
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    en.record()
    torch.cuda.synchronize()
    t=st.elapsed_time(en)/10*1000
    print(f"  M={_m} N={_n} K={_k}: {t:.1f}us")
    del _A,_Bw,_Bws
torch.cuda.empty_cache()

# Also build fused quant+CK for shape 6
from torch.utils.cpp_extension import load_inline
_hip_ck = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
extern "C" __global__ void fused_quant_shuffle(
    const unsigned short* __restrict__ A, unsigned char* __restrict__ A_q,
    unsigned char* __restrict__ A_scale_sh, int M, int K, int K_half, int K_groups, int M_pad) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = M_pad * K_groups;
    if (gid >= total_groups) return;
    int m = gid / K_groups; int g = gid % K_groups;
    float vals[32]; float amax = 0.0f;
    if (m < M) {
        const uint4* row4 = (const uint4*)(A + m * K + g * 32);
        #pragma unroll
        for (int v=0;v<4;v++){uint4 c=row4[v];unsigned int w[4]={c.x,c.y,c.z,c.w};
            #pragma unroll
            for(int j=0;j<4;j++){float lo=__uint_as_float((w[j]&0xFFFFu)<<16);float hi=__uint_as_float(w[j]&0xFFFF0000u);
                vals[v*8+j*2]=lo;vals[v*8+j*2+1]=hi;amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi)));}}
    } else { for(int i=0;i<32;i++) vals[i]=0.0f; }
    unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
    int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
    int m0=m/32;int m1=(m&31)/16;int m2=m&15;int g0=g/8;int g1=(g&7)/4;int g2=g&3;
    int kg8=K_groups/8;
    A_scale_sh[m0*(kg8*256)+g0*256+g2*64+m2*4+g1*2+m1]=(unsigned char)sb;
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
    uint4* out4=(uint4*)(A_q+m*K_half+g*16);*out4=*((uint4*)packed);
}
struct __attribute__((packed)) KArgs {
    void* ptr_D;char _p0[8];void* ptr_C;char _p1[8];void* ptr_A;char _p2[8];void* ptr_B;char _p3[8];
    float alpha;char _p4[12];float beta;char _p5[12];
    unsigned int stride_D0;char _p6[12];unsigned int stride_D1;char _p7[12];
    unsigned int stride_C0;char _p8[12];unsigned int stride_C1;char _p9[12];
    unsigned int stride_A0;char _p10[12];unsigned int stride_A1;char _p11[12];
    unsigned int stride_B0;char _p12[12];unsigned int stride_B1;char _p13[12];
    unsigned int M;char _p14[12];unsigned int N;char _p15[12];unsigned int K;char _p16[12];
    void* ptr_ScaleA;char _p17[8];void* ptr_ScaleB;char _p18[8];
    unsigned int stride_ScaleA0;char _p19[12];unsigned int stride_ScaleA1;char _p20[12];
    unsigned int stride_ScaleB0;char _p21[12];unsigned int stride_ScaleB1;char _p22[12];
    int log2_k_split;
};
static hipModule_t g_ck_mod=nullptr;static hipFunction_t g_ck_fn=nullptr;
static torch::Tensor g_aq,g_ash,g_out;static int g_M_pad=0,g_K=0,g_N=0;
bool init_ck(const std::string& co,const std::string& fn){
    if(g_ck_fn)return true;
    if(hipModuleLoad(&g_ck_mod,co.c_str())!=hipSuccess)return false;
    if(hipModuleGetFunction(&g_ck_fn,g_ck_mod,fn.c_str())!=hipSuccess){g_ck_mod=nullptr;return false;}
    return true;
}
torch::Tensor quant_and_gemm(torch::Tensor A,torch::Tensor Bs,torch::Tensor Bss,int Nv){
    if(!g_ck_fn)return torch::Tensor();
    int M=A.size(0),Ke=A.size(1),Mp=((M+31)/32)*32,Kh=Ke/2,Kg=Ke/32;
    if(Mp!=g_M_pad||Ke!=g_K){g_aq=torch::empty({Mp,Kh},torch::dtype(torch::kUInt8).device(A.device()));
        g_ash=torch::empty({Mp,Kg},torch::dtype(torch::kUInt8).device(A.device()));g_M_pad=Mp;g_K=Ke;}
    if(Nv!=g_N||Mp!=g_M_pad){g_out=torch::empty({Mp,Nv},torch::dtype(torch::kBFloat16).device(A.device()));g_N=Nv;}
    int tg=Mp*Kg;fused_quant_shuffle<<<(tg+63)/64,64,0,0>>>((const unsigned short*)A.data_ptr(),
        g_aq.data_ptr<unsigned char>(),g_ash.data_ptr<unsigned char>(),M,Ke,Kh,Kg,Mp);
    KArgs ka;memset(&ka,0,sizeof(ka));
    ka.ptr_D=g_out.data_ptr();ka.ptr_C=g_out.data_ptr();
    ka.ptr_A=g_aq.data_ptr<unsigned char>();ka.ptr_B=Bs.data_ptr();
    ka.alpha=1.0f;ka.beta=0.0f;ka.stride_D0=Nv;ka.stride_D1=1;ka.stride_C0=Nv;ka.stride_C1=1;
    ka.stride_A0=Ke;ka.stride_A1=1;ka.stride_B0=Bs.stride(0)*2;ka.stride_B1=1;
    ka.M=Mp;ka.N=Nv;ka.K=Ke;ka.ptr_ScaleA=g_ash.data_ptr<unsigned char>();ka.ptr_ScaleB=Bss.data_ptr();
    ka.stride_ScaleA0=Kg;ka.stride_ScaleA1=1;ka.stride_ScaleB0=Bss.stride(0);ka.stride_ScaleB1=1;
    ka.log2_k_split=0;size_t asz=sizeof(ka);
    void* cfg[]={(void*)0x01,&ka,(void*)0x02,&asz,(void*)0x03};
    hipModuleLaunchKernel(g_ck_fn,(Nv+127)/128,(Mp+31)/32,1,256,1,1,0,0,nullptr,cfg);
    return g_out.slice(0,0,M);
}
"""
_cpp_ck=r"""
#include <torch/extension.h>
bool init_ck(const std::string&,const std::string&);
torch::Tensor quant_and_gemm(torch::Tensor,torch::Tensor,torch::Tensor,int);
"""
_ck_mod=None
try:
    _ck_mod=load_inline(name='ck554',cpp_sources=_cpp_ck,cuda_sources=_hip_ck,
        functions=['init_ck','quant_and_gemm'],verbose=False,
        extra_cuda_cflags=['-O3','-w','-mcumode','--offload-arch=gfx950'])
    _co="/home/runner/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.co"
    _fn="_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
    if _ck_mod.init_ck(_co,_fn):
        _w=torch.randn(256,1536,dtype=torch.bfloat16,device="cuda")
        _ck_mod.quant_and_gemm(_w,torch.zeros(3072,768,dtype=torch.uint8,device="cuda"),
            torch.zeros(3072,48,dtype=torch.uint8,device="cuda"),3072)
        torch.cuda.synchronize();del _w;print("CK OK")
    else:_ck_mod=None
except Exception as e:_ck_mod=None;print(f"CK fail: {e}")
torch.cuda.empty_cache()

_ps_ck=None;_ps_cw=None;_ps_cs=None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    if _ck_mod is not None and m==256 and k==1536:
        return _ck_mod.quant_and_gemm(A,B_shuffle,B_scale_sh,n)
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
