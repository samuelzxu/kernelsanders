#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#384: Measure EXACTLY where time goes in the hipBLASLt pipeline.
Use hipEvent inside C++ to measure each step independently.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

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
#define CHRTC(x) do{hiprtcResult r=(x);if(r!=HIPRTC_SUCCESS)printf("R%d\n",(int)r);}while(0)
#define CHIP(x) do{hipError_t e=(x);if(e)printf("H%d@%d\n",(int)e,__LINE__);}while(0)
static hipModule_t g_qmod=nullptr;static hipFunction_t g_qfn=nullptr;
static const char* qsrc = R"(
extern "C" __global__ void mxfp4_quant(const unsigned short* __restrict__ A,unsigned char* __restrict__ Aq,unsigned char* __restrict__ As,int M,int K,int Kh,int Kg){
int gid=blockIdx.x*blockDim.x+threadIdx.x;if(gid>=M*Kg)return;
int m=gid/Kg,g=gid%Kg;
auto bf2f=[](unsigned short b)->float{unsigned int f=(unsigned int)b<<16;return *reinterpret_cast<float*>(&f);};
float amax=0;const unsigned short*row=A+m*K+g*32;
for(int i=0;i<32;i++){float v=bf2f(row[i]);float a=v<0?-v:v;if(a>amax)amax=a;}
union{float f;unsigned int i;}u;u.f=amax;
unsigned int ar=(u.i+0x200000u)&0xFF800000u;u.i=ar;
int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
As[m*Kg+g]=(unsigned char)sb;
int qe=127-su;if(qe<1)qe=0;if(qe>254)qe=254;
union{float f2;unsigned int i2;}qs;qs.i2=(unsigned int)qe<<23;
float qsc=qs.f2;if(amax==0)qsc=0;
const unsigned int dmi=149<<23;union{unsigned int di;float df;}dm;dm.di=dmi;
const int vta=((1-127)<<23)+(1<<21)-1;
unsigned char pk[16];
for(int i=0;i<16;i++){
auto fp4=[&](unsigned short b)->unsigned char{
float qf=bf2f(b)*qsc;unsigned int qx;{union{float f;unsigned int i;}t;t.f=qf;qx=t.i;}
unsigned int s=qx&0x80000000u;qx^=s;
float qp;{union{unsigned int i;float f;}t;t.i=qx;qp=t.f;}
unsigned char r;
if(qp>=6.0f)r=7;else if(qp<1.0f){float dx=qp+dm.df;unsigned int di;{union{float f;unsigned int i;}t;t.f=dx;di=t.i;}di-=dmi;r=(unsigned char)(di&0xFF);}
else{unsigned int nx=qx;unsigned int mo=(nx>>22)&1;nx=(unsigned int)((int)nx+vta);nx+=mo;nx>>=22;r=(unsigned char)(nx&0xFF);}
return(unsigned char)((r&7)|((s>>28)&8));};
pk[i]=fp4(row[2*i])|(fp4(row[2*i+1])<<4);}
unsigned char*o=Aq+m*Kh+g*16;for(int i=0;i<16;i++)o[i]=pk[i];}
)";
static hipblasLtHandle_t g_h=nullptr;
static hipblasLtMatmulDesc_t g_md=nullptr;
static hipblasLtMatrixLayout_t g_lA,g_lB,g_lC;
static hipblasLtMatmulHeuristicResult_t g_hr;

void init(int M,int N,int K,torch::Tensor As_buf,torch::Tensor Bs_buf){
    if(!g_qfn){hiprtcProgram p;CHRTC(hiprtcCreateProgram(&p,qsrc,"q.hip",0,0,0));
    const char*o[]={"--offload-arch=gfx950","-O3"};CHRTC(hiprtcCompileProgram(p,2,o));
    size_t sz;CHRTC(hiprtcGetCodeSize(p,&sz));char*c=new char[sz];CHRTC(hiprtcGetCode(p,c));
    CHIP(hipModuleLoadData(&g_qmod,c));CHIP(hipModuleGetFunction(&g_qfn,g_qmod,"mxfp4_quant"));
    delete[]c;hiprtcDestroyProgram(&p);}
    if(!g_h)hipblasLtCreate(&g_h);
    if(g_md){hipblasLtMatrixLayoutDestroy(g_lA);hipblasLtMatrixLayoutDestroy(g_lB);
    hipblasLtMatrixLayoutDestroy(g_lC);hipblasLtMatmulDescDestroy(g_md);}
    CK(hipblasLtMatmulDescCreate(&g_md,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    hipblasOperation_t opT=HIPBLAS_OP_T,opN=HIPBLAS_OP_N;
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT)));
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN)));
    hipblasLtMatmulMatrixScale_t sm=HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm)));
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm)));
    void*bp=Bs_buf.data_ptr();void*ap=As_buf.data_ptr();
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,&bp,sizeof(bp)));
    CK(hipblasLtMatmulDescSetAttribute(g_md,HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,&ap,sizeof(ap)));
    hipDataType fp4t=(hipDataType)33;
    CK(hipblasLtMatrixLayoutCreate(&g_lA,fp4t,K,N,K));
    CK(hipblasLtMatrixLayoutCreate(&g_lB,fp4t,K,M,K));
    CK(hipblasLtMatrixLayoutCreate(&g_lC,HIP_R_16BF,N,M,N));
    hipblasLtMatmulPreference_t pf;CK(hipblasLtMatmulPreferenceCreate(&pf));
    size_t ws=0;CK(hipblasLtMatmulPreferenceSetAttribute(pf,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
    int cnt=0;CK(hipblasLtMatmulAlgoGetHeuristic(g_h,g_md,g_lA,g_lB,g_lC,g_lC,pf,1,&g_hr,&cnt));
    hipblasLtMatmulPreferenceDestroy(pf);
    printf("INIT cnt=%d\n",cnt);
}

std::string measure(torch::Tensor A,torch::Tensor Aq,torch::Tensor As,
                    torch::Tensor Bq,torch::Tensor out,int M,int N,int K){
    hipEvent_t e0,e1,e2,e3;
    hipEventCreate(&e0);hipEventCreate(&e1);hipEventCreate(&e2);hipEventCreate(&e3);

    // Warmup
    int Kh=K/2,Kg=K/32,tot=M*Kg;
    struct{void*A;void*Aq;void*As;int M;int K;int Kh;int Kg;}args;
    args.A=A.data_ptr();args.Aq=Aq.data_ptr();args.As=As.data_ptr();
    args.M=M;args.K=K;args.Kh=Kh;args.Kg=Kg;
    size_t sz=sizeof(args);void*cfg[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,&args,HIP_LAUNCH_PARAM_BUFFER_SIZE,&sz,HIP_LAUNCH_PARAM_END};
    float alpha=1,beta=0;
    for(int i=0;i<5;i++){
        CHIP(hipModuleLaunchKernel(g_qfn,(tot+255)/256,1,1,256,1,1,0,0,nullptr,cfg));
        CK(hipblasLtMatmul(g_h,g_md,&alpha,Bq.data_ptr(),g_lA,Aq.data_ptr(),g_lB,&beta,out.data_ptr(),g_lC,out.data_ptr(),g_lC,&g_hr.algo,nullptr,0,0));
    }
    hipDeviceSynchronize();

    // Measure 100 iterations
    hipEventRecord(e0);
    for(int i=0;i<100;i++){
        CHIP(hipModuleLaunchKernel(g_qfn,(tot+255)/256,1,1,256,1,1,0,0,nullptr,cfg));
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float q_ms; hipEventElapsedTime(&q_ms,e0,e1);

    hipEventRecord(e0);
    for(int i=0;i<100;i++){
        CK(hipblasLtMatmul(g_h,g_md,&alpha,Bq.data_ptr(),g_lA,Aq.data_ptr(),g_lB,&beta,out.data_ptr(),g_lC,out.data_ptr(),g_lC,&g_hr.algo,nullptr,0,0));
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float g_ms; hipEventElapsedTime(&g_ms,e0,e1);

    hipEventRecord(e0);
    for(int i=0;i<100;i++){
        CHIP(hipModuleLaunchKernel(g_qfn,(tot+255)/256,1,1,256,1,1,0,0,nullptr,cfg));
        CK(hipblasLtMatmul(g_h,g_md,&alpha,Bq.data_ptr(),g_lA,Aq.data_ptr(),g_lB,&beta,out.data_ptr(),g_lC,out.data_ptr(),g_lC,&g_hr.algo,nullptr,0,0));
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float t_ms; hipEventElapsedTime(&t_ms,e0,e1);

    char buf[256];
    snprintf(buf,sizeof(buf),"quant_only=%.1fus gemm_only=%.1fus both=%.1fus gap=%.1fus",
             q_ms*10,g_ms*10,t_ms*10,t_ms*10-q_ms*10-g_ms*10);

    hipEventDestroy(e0);hipEventDestroy(e1);hipEventDestroy(e2);hipEventDestroy(e3);
    return std::string(buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("init",&init);m.def("measure",&measure);}
"""
mod=load_inline(name="hblt_meas",cpp_sources=[cpp],extra_include_paths=["/opt/rocm/include"],
extra_ldflags=["-L/opt/rocm/lib","-lhipblaslt","-lhiprtc","-lamdhip64"],verbose=False,is_python_module=True)

M,N,K=256,3072,1536
_aq=torch.empty(M,K//2,dtype=torch.uint8,device="cuda")
_as=torch.empty(M,K//32,dtype=torch.uint8,device="cuda")
_out=torch.empty(M,N,dtype=torch.bfloat16,device="cuda")
_bq=torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
_bs=torch.zeros(N,K//32,dtype=torch.uint8,device="cuda")
_a=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
mod.init(M,N,K,_as,_bs)
result=mod.measure(_a,_aq,_as,_bq,_out,M,N,K)
print(f"K=1536 M=256: {result}")

# Also measure K=2048 M=64
M2,N2,K2=64,7168,2048
_aq2=torch.empty(M2,K2//2,dtype=torch.uint8,device="cuda")
_as2=torch.empty(M2,K2//32,dtype=torch.uint8,device="cuda")
_out2=torch.empty(M2,N2,dtype=torch.bfloat16,device="cuda")
_bq2=torch.zeros(N2,K2//2,dtype=torch.uint8,device="cuda")
_bs2=torch.zeros(N2,K2//32,dtype=torch.uint8,device="cuda")
_a2=torch.randn(M2,K2,dtype=torch.bfloat16,device="cuda")
mod.init(M2,N2,K2,_as2,_bs2)
r2=mod.measure(_a2,_aq2,_as2,_bq2,_out2,M2,N2,K2)
print(f"K=2048 M=64: {r2}")

torch.cuda.empty_cache()

# Standard preshuffle
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()
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
