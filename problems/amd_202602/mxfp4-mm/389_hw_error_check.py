#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#389: Measure EXACT error of HW quant + hipBLASLt vs reference.
Build the full pipeline and compare outputs.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
from reference import generate_input

def e8m0_unshuffle(s,N,Kg):
    sm,sn=s.shape;t=s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous()
    return t.view(sm,sn)[:N,:Kg].contiguous()

cpp = r"""
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#define CK(x) do{hipblasStatus_t s=(x);if(s)printf("B%d@%d\n",(int)s,__LINE__);}while(0)
#define CHIP(x) do{hipError_t e=(x);if(e)printf("H%d@%d\n",(int)e,__LINE__);}while(0)
static hipModule_t g_qmod=nullptr;static hipFunction_t g_qfn=nullptr;
static const char* qsrc = R"(
typedef __attribute__((ext_vector_type(2))) unsigned short v2u16;
extern "C" __global__ void mxfp4_quant_hw(const unsigned int* __restrict__ A,unsigned char* __restrict__ Aq,unsigned char* __restrict__ As,int M,int K,int Kh,int Kg){
int gid=blockIdx.x*blockDim.x+threadIdx.x;if(gid>=M*Kg)return;
int m=gid/Kg,g=gid%Kg;
const unsigned int*row=A+(m*K+g*32)/2;
unsigned int v[16];const uint4*r4=(const uint4*)row;
uint4 l0=r4[0];v[0]=l0.x;v[1]=l0.y;v[2]=l0.z;v[3]=l0.w;
uint4 l1=r4[1];v[4]=l1.x;v[5]=l1.y;v[6]=l1.z;v[7]=l1.w;
uint4 l2=r4[2];v[8]=l2.x;v[9]=l2.y;v[10]=l2.z;v[11]=l2.w;
uint4 l3=r4[3];v[12]=l3.x;v[13]=l3.y;v[14]=l3.z;v[15]=l3.w;
float amax=0;
for(int i=0;i<16;i++){unsigned int lo=v[i]&0xFFFF,hi=v[i]>>16;lo<<=16;hi<<=16;
float f0=*reinterpret_cast<float*>(&lo),f1=*reinterpret_cast<float*>(&hi);
float a0=f0<0?-f0:f0,a1=f1<0?-f1:f1;if(a0>amax)amax=a0;if(a1>amax)amax=a1;}
union{float f;unsigned int i;}u;u.f=amax;
unsigned int ar=(u.i+0x200000u)&0xFF800000u;u.i=ar;
int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
As[m*Kg+g]=(unsigned char)sb;
// The intrinsic takes the scale EXPONENT as float: it computes bf16 * 2^(-scale)
// So pass su (unbiased exponent) directly
// Try the original inv_scale approach but check if scale is correct
union{float f;unsigned int i;}sc; sc.i=((unsigned int)(sb>0?sb:1))<<23;
float inv = 1.0f/sc.f; if(amax==0)inv=0.0f;
unsigned int result=0;
#pragma unroll
for(int b=0;b<16;b++){
v2u16 bf=*reinterpret_cast<v2u16*>(&v[b]);
switch(b&3){
case 0:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,inv,0);break;
case 1:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,inv,1);break;
case 2:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,inv,2);break;
case 3:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,inv,3);break;}
if((b&3)==3){((unsigned int*)(Aq+m*Kh+g*16))[b/4]=result;result=0;}}}
)";
static hipblasLtHandle_t g_h=nullptr;
static hipblasLtMatmulDesc_t g_md=nullptr;
static hipblasLtMatrixLayout_t g_lA,g_lB,g_lC;
static hipblasLtMatmulHeuristicResult_t g_hr;
static struct{void*A;void*Aq;void*As;int M;int K;int Kh;int Kg;}g_qa;
static size_t g_qs;static void*g_qc[5];static int g_qb;

void setup(int M,int N,int K,torch::Tensor As,torch::Tensor Bs,torch::Tensor Aq,torch::Tensor Bq,torch::Tensor out){
    if(!g_qfn){hiprtcProgram p;
    hiprtcCreateProgram(&p,qsrc,"q.hip",0,0,0);
    const char*o[]={"--offload-arch=gfx950","-O3"};
    hiprtcResult r=hiprtcCompileProgram(p,2,o);
    if(r!=HIPRTC_SUCCESS){size_t l;hiprtcGetProgramLogSize(p,&l);char*lg=new char[l];hiprtcGetProgramLog(p,lg);printf("CE:%s\n",lg);delete[]lg;return;}
    size_t sz;hiprtcGetCodeSize(p,&sz);char*c=new char[sz];hiprtcGetCode(p,c);
    CHIP(hipModuleLoadData(&g_qmod,c));CHIP(hipModuleGetFunction(&g_qfn,g_qmod,"mxfp4_quant_hw"));
    delete[]c;hiprtcDestroyProgram(&p);}
    if(!g_h)hipblasLtCreate(&g_h);
    if(g_md){hipblasLtMatrixLayoutDestroy(g_lA);hipblasLtMatrixLayoutDestroy(g_lB);hipblasLtMatrixLayoutDestroy(g_lC);hipblasLtMatmulDescDestroy(g_md);}
    CK(hipblasLtMatmulDescCreate(&g_md,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    hipblasOperation_t opT=HIPBLAS_OP_T,opN=HIPBLAS_OP_N;
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT)));
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN)));
    hipblasLtMatmulMatrixScale_t sm=HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm)));
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm)));
    void*bp=Bs.data_ptr();void*ap=As.data_ptr();
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,&bp,sizeof(bp)));
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,&ap,sizeof(ap)));
    hipDataType fp4t=(hipDataType)33;
    CK(hipblasLtMatrixLayoutCreate(&g_lA,fp4t,K,N,K));CK(hipblasLtMatrixLayoutCreate(&g_lB,fp4t,K,M,K));
    CK(hipblasLtMatrixLayoutCreate(&g_lC,HIP_R_16BF,N,M,N));
    hipblasLtMatmulPreference_t pf;CK(hipblasLtMatmulPreferenceCreate(&pf));
    size_t ws=0;CK(hipblasLtMatmulPreferenceSetAttribute(pf,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
    int cnt=0;CK(hipblasLtMatmulAlgoGetHeuristic(g_h,g_md,g_lA,g_lB,g_lC,g_lC,pf,1,&g_hr,&cnt));
    hipblasLtMatmulPreferenceDestroy(pf);
    g_qa.Aq=Aq.data_ptr();g_qa.As=As.data_ptr();g_qa.M=M;g_qa.K=K;g_qa.Kh=K/2;g_qa.Kg=K/32;
    g_qs=sizeof(g_qa);g_qc[0]=HIP_LAUNCH_PARAM_BUFFER_POINTER;g_qc[1]=&g_qa;
    g_qc[2]=HIP_LAUNCH_PARAM_BUFFER_SIZE;g_qc[3]=&g_qs;g_qc[4]=HIP_LAUNCH_PARAM_END;
    g_qb=(M*(K/32)+255)/256;
}
torch::Tensor run(torch::Tensor A,torch::Tensor Aq,torch::Tensor Bq,torch::Tensor out){
    g_qa.A=A.data_ptr();
    CHIP(hipModuleLaunchKernel(g_qfn,g_qb,1,1,256,1,1,0,0,nullptr,g_qc));
    float alpha=1,beta=0;
    CK(hipblasLtMatmul(g_h,g_md,&alpha,Bq.data_ptr(),g_lA,Aq.data_ptr(),g_lB,&beta,out.data_ptr(),g_lC,out.data_ptr(),g_lC,&g_hr.algo,nullptr,0,0));
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("setup",&setup);m.def("run",&run);}
"""
mod=load_inline(name="hblt_ec",cpp_sources=[cpp],extra_include_paths=["/opt/rocm/include"],
extra_ldflags=["-L/opt/rocm/lib","-lhipblaslt","-lhiprtc","-lamdhip64"],verbose=False,is_python_module=True)

# Test with actual benchmark data
data=generate_input(m=256,n=3072,k=1536,seed=7856)
A,B,B_q,B_shuffle,B_scale_sh=data
m,k=A.shape;n=B.shape[0]

# Reference
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=3072-K=1536":{"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
ref=gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)

# HW quant + hipBLASLt
_aq=torch.empty(m,k//2,dtype=torch.uint8,device="cuda")
_as=torch.empty(m,k//32,dtype=torch.uint8,device="cuda")
_out=torch.empty(m,n,dtype=torch.bfloat16,device="cuda")
bs_raw=e8m0_unshuffle(B_scale_sh,n,k//32)
mod.setup(m,n,k,_as,bs_raw,_aq,B_q,_out)
torch.cuda.synchronize()
hw_out=mod.run(A,_aq,B_q,_out)
torch.cuda.synchronize()

diff=(ref-hw_out).abs()
print(f"HW quant + hipBLASLt vs reference:")
print(f"  max_err={diff.max().item():.1f}")
print(f"  mean_err={diff.mean().item():.3f}")
print(f"  >1.0 count: {(diff>1.0).sum().item()}")
print(f"  >0.5 count: {(diff>0.5).sum().item()}")
print(f"  ref range: [{ref.min().item():.0f}, {ref.max().item():.0f}]")

# Also test SW quant + hipBLASLt for comparison
from aiter.ops.triton.quant import dynamic_mxfp4_quant
Aq_sw,As_sw=dynamic_mxfp4_quant(A)
As_sw=As_sw.contiguous()
_aq2=torch.empty(m,k//2,dtype=torch.uint8,device="cuda")
_as2=torch.empty(m,k//32,dtype=torch.uint8,device="cuda")
_out2=torch.empty(m,n,dtype=torch.bfloat16,device="cuda")
mod.setup(m,n,k,_as2,bs_raw,_aq2,B_q,_out2)
# Copy SW quant results into hipBLASLt buffers
_aq2.copy_(Aq_sw)
_as2.copy_(As_sw)
# Run GEMM only (skip quant kernel)
g_out=torch.empty(m,n,dtype=torch.bfloat16,device="cuda")
# Can't easily skip quant in C++, so just compare HW vs SW quant bytes
hw_aq=_aq.clone()  # contains HW quant result
torch.cuda.synchronize()
aq_diff=(Aq_sw!=hw_aq).sum().item()
print(f"\nHW vs SW quant A_q diff: {aq_diff}/{Aq_sw.numel()} ({100*aq_diff/Aq_sw.numel():.1f}%)")

del A,B,B_q,B_shuffle,B_scale_sh,ref,hw_out;torch.cuda.empty_cache()

# Standard preshuffle for correctness
_cfgs2={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
for _sk,_cfg in _cfgs2.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_ck=None;_cw2=None;_cs2=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck,_cw2,_cs2
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ck:
        _ck=dp;_cw2=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _cs2=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_cw2,_cs2,prequant=True,dtype=torch.bfloat16)
