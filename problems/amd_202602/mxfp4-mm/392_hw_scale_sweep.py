#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#392: Sweep ALL possible scale interpretations for the HW quant intrinsic.
Test each against dynamic_mxfp4_quant output to find the correct one.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Build a kernel that tries MULTIPLE scale values and reports which matches
cpp = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#define CHIP(x) do{hipError_t e=(x);if(e)printf("H%d@%d\\n",(int)e,__LINE__);}while(0)
static hipModule_t g_m=nullptr; static hipFunction_t g_f=nullptr;
static const char* src = R"KERNEL(
typedef __attribute__((ext_vector_type(2))) unsigned short v2u16;
extern "C" __global__ void test_hw_quant(
    const unsigned int* __restrict__ A,
    unsigned char* __restrict__ out0, // scale = inv_scale (1/2^su)
    unsigned char* __restrict__ out1, // scale = -su (float)
    unsigned char* __restrict__ out2, // scale = su (float)
    unsigned char* __restrict__ out3, // scale = sb (float, biased)
    unsigned char* __restrict__ out4, // scale = su+2
    unsigned char* __restrict__ out5, // scale = -su-2
    unsigned char* __restrict__ As,
    int M, int K, int Kh, int Kg
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= M * Kg) return;
    int m = gid / Kg, g = gid % Kg;
    const unsigned int* row = A + (m*K + g*32)/2;
    unsigned int v[16]; const uint4* r4 = (const uint4*)row;
    uint4 l0=r4[0];v[0]=l0.x;v[1]=l0.y;v[2]=l0.z;v[3]=l0.w;
    uint4 l1=r4[1];v[4]=l1.x;v[5]=l1.y;v[6]=l1.z;v[7]=l1.w;
    uint4 l2=r4[2];v[8]=l2.x;v[9]=l2.y;v[10]=l2.z;v[11]=l2.w;
    uint4 l3=r4[3];v[12]=l3.x;v[13]=l3.y;v[14]=l3.z;v[15]=l3.w;
    float amax=0;for(int i=0;i<16;i++){unsigned int lo=v[i]&0xFFFF,hi=v[i]>>16;lo<<=16;hi<<=16;
    float f0=*reinterpret_cast<float*>(&lo),f1=*reinterpret_cast<float*>(&hi);
    float a0=f0<0?-f0:f0,a1=f1<0?-f1:f1;if(a0>amax)amax=a0;if(a1>amax)amax=a1;}
    union{float f;unsigned int i;}u;u.f=amax;unsigned int ar=(u.i+0x200000u)&0xFF800000u;u.i=ar;
    int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
    As[m*Kg+g]=(unsigned char)sb;

    // Compute different scale values
    union{float f;unsigned int i;}sc;sc.i=((unsigned int)(sb>0?sb:1))<<23;
    float inv_scale = 1.0f/sc.f; if(amax==0)inv_scale=0.0f;
    float scales[6] = {inv_scale, (float)(-su), (float)su, (float)sb, (float)(su+2), (float)(-su-2)};
    unsigned char* outs[6] = {out0,out1,out2,out3,out4,out5};

    for(int s=0;s<6;s++){
        float scale = scales[s];
        unsigned int result = 0;
        for(int b=0;b<16;b++){
            v2u16 bf=*reinterpret_cast<v2u16*>(&v[b]);
            switch(b&3){
            case 0:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,scale,0);break;
            case 1:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,scale,1);break;
            case 2:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,scale,2);break;
            case 3:result=__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(result,bf,scale,3);break;}
            if((b&3)==3){((unsigned int*)(outs[s]+m*Kh+g*16))[b/4]=result;result=0;}
        }
    }
}
)KERNEL";
void init(){hiprtcProgram p;hiprtcCreateProgram(&p,src,"q.hip",0,0,0);
const char*o[]={"--offload-arch=gfx950","-O3"};hiprtcCompileProgram(p,2,o);
size_t sz;hiprtcGetCodeSize(p,&sz);char*c=new char[sz];hiprtcGetCode(p,c);
CHIP(hipModuleLoadData(&g_m,c));CHIP(hipModuleGetFunction(&g_f,g_m,"test_hw_quant"));
delete[]c;hiprtcDestroyProgram(&p);printf("OK\\n");}
void run(torch::Tensor A,torch::Tensor o0,torch::Tensor o1,torch::Tensor o2,
         torch::Tensor o3,torch::Tensor o4,torch::Tensor o5,torch::Tensor As){
int M=A.size(0),K=A.size(1),Kh=K/2,Kg=K/32,tot=M*Kg;
struct{void*A;void*o0;void*o1;void*o2;void*o3;void*o4;void*o5;void*As;int M;int K;int Kh;int Kg;}args;
args.A=A.data_ptr();args.o0=o0.data_ptr();args.o1=o1.data_ptr();args.o2=o2.data_ptr();
args.o3=o3.data_ptr();args.o4=o4.data_ptr();args.o5=o5.data_ptr();args.As=As.data_ptr();
args.M=M;args.K=K;args.Kh=Kh;args.Kg=Kg;
size_t sz=sizeof(args);void*cfg[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,&args,HIP_LAUNCH_PARAM_BUFFER_SIZE,&sz,HIP_LAUNCH_PARAM_END};
CHIP(hipModuleLaunchKernel(g_f,(tot+255)/256,1,1,256,1,1,0,0,nullptr,cfg));
CHIP(hipDeviceSynchronize());}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("init",&init);m.def("run",&run);}
"""
mod=load_inline(name="hw_sweep",cpp_sources=[cpp],extra_include_paths=["/opt/rocm/include"],
extra_ldflags=["-L/opt/rocm/lib","-lhiprtc","-lamdhip64"],verbose=False,is_python_module=True)
mod.init()

M,K=32,512
A=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
Aq_ref,As_ref=dynamic_mxfp4_quant(A)

outs=[torch.empty(M,K//2,dtype=torch.uint8,device="cuda") for _ in range(6)]
As_hw=torch.empty(M,K//32,dtype=torch.uint8,device="cuda")
mod.run(A,*outs,As_hw)

labels=["inv_scale","neg_su","su","sb","su+2","neg_su_m2"]
for i,(label,o) in enumerate(zip(labels,outs)):
    diff=(Aq_ref!=o).sum().item()
    pct=100*diff/Aq_ref.numel()
    print(f"  {label}: {diff}/{Aq_ref.numel()} diff ({pct:.1f}%)")

# Also check scale diff
sdiff=(As_ref.contiguous()!=As_hw).sum().item()
print(f"  Scale diff: {sdiff}/{As_ref.numel()}")

# Standard kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
_ck=None;_cw=None;_cs=None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck,_cw,_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]
    dp=B_shuffle.data_ptr()
    if dp!=_ck:
        _ck=dp;_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_cw,_cs,prequant=True,dtype=torch.bfloat16)
