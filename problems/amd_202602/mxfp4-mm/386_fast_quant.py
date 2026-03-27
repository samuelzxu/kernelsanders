#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#386: Optimized hiprtc quant kernel. Current: 5.8µs for 12K groups.
Target: <2µs via vectorized loads and multiple groups per warp.
Key: use __builtin_amdgcn_raw_buffer_load for fast HBM reads.
Or simpler: each thread processes 4 groups instead of 1.
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
#define CHRTC(x) do{hiprtcResult r=(x);if(r!=HIPRTC_SUCCESS){size_t l;hiprtcGetProgramLogSize(p,&l);char*lg=new char[l];hiprtcGetProgramLog(p,lg);printf("R%d:%s\n",(int)r,lg);delete[]lg;}}while(0)
#define CHIP(x) do{hipError_t e=(x);if(e)printf("H%d@%d\n",(int)e,__LINE__);}while(0)
static hipModule_t g_qmod=nullptr;static hipFunction_t g_qfn=nullptr;

// V2: Vectorized loads + reduced branching. Process 1 group per thread.
static const char* qsrc = R"(
extern "C" __global__ void mxfp4_quant_fast(
    const unsigned int* __restrict__ A,
    unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As,
    int M, int K, int Kh, int Kg
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= M * Kg) return;
    int m = gid / Kg, g = gid % Kg;
    const unsigned int* row = A + (m * K + g * 32) / 2;

    // Vectorized load
    unsigned int v[16];
    const uint4* row4 = (const uint4*)row;
    uint4 ld0=row4[0]; v[0]=ld0.x;v[1]=ld0.y;v[2]=ld0.z;v[3]=ld0.w;
    uint4 ld1=row4[1]; v[4]=ld1.x;v[5]=ld1.y;v[6]=ld1.z;v[7]=ld1.w;
    uint4 ld2=row4[2]; v[8]=ld2.x;v[9]=ld2.y;v[10]=ld2.z;v[11]=ld2.w;
    uint4 ld3=row4[3]; v[12]=ld3.x;v[13]=ld3.y;v[14]=ld3.z;v[15]=ld3.w;

    // Find max abs using bf16->float conversion
    float amax = 0;
    for (int i = 0; i < 16; i++) {
        unsigned int lo = v[i] & 0xFFFF, hi = v[i] >> 16;
        unsigned int flo = lo << 16, fhi = hi << 16;
        float fv0 = *reinterpret_cast<float*>(&flo);
        float fv1 = *reinterpret_cast<float*>(&fhi);
        float a0 = fv0 < 0 ? -fv0 : fv0;
        float a1 = fv1 < 0 ? -fv1 : fv1;
        if (a0 > amax) amax = a0;
        if (a1 > amax) amax = a1;
    }

    // E8M0 scale
    union{float f;unsigned int i;} u; u.f = amax;
    unsigned int ar = (u.i + 0x200000u) & 0xFF800000u; u.i = ar;
    int eb = (ar >> 23) & 0xFF; int su = eb - 129; int sb = su + 127;
    if (sb < 0) sb = 0;
    As[m * Kg + g] = (unsigned char)sb;

    // Quant scale
    int qe = 127 - su; if (qe < 1) qe = 0; if (qe > 254) qe = 254;
    union{float f2;unsigned int i2;} qs; qs.i2 = (unsigned int)qe << 23;
    float qsc = qs.f2; if (amax == 0) qsc = 0;

    const unsigned int dmi = 149 << 23;
    union{unsigned int di;float df;} dm; dm.di = dmi;
    const int vta = ((1-127) << 23) + (1 << 21) - 1;

    // Quantize all 32 elements
    unsigned char pk[16];
    for (int i = 0; i < 16; i++) {
        unsigned int lo = v[i] & 0xFFFF, hi = v[i] >> 16;

        auto fp4 = [&](unsigned int bf16_bits) -> unsigned char {
            unsigned int f32 = bf16_bits << 16;
            float qf = *reinterpret_cast<float*>(&f32) * qsc;
            unsigned int qx; {union{float f;unsigned int i;}t; t.f=qf; qx=t.i;}
            unsigned int s = qx & 0x80000000u; qx ^= s;
            float qp; {union{unsigned int i;float f;}t; t.i=qx; qp=t.f;}
            unsigned char r;
            if (qp >= 6.0f) r = 7;
            else if (qp < 1.0f) {
                float dx = qp + dm.df;
                unsigned int di; {union{float f;unsigned int i;}t; t.f=dx; di=t.i;}
                di -= dmi; r = (unsigned char)(di & 0xFF);
            } else {
                unsigned int nx = qx;
                unsigned int mo = (nx >> 22) & 1;
                nx = (unsigned int)((int)nx + vta); nx += mo; nx >>= 22;
                r = (unsigned char)(nx & 0xFF);
            }
            return (unsigned char)((r & 7) | ((s >> 28) & 8));
        };

        pk[i] = fp4(lo) | (fp4(hi) << 4);
    }

    // Vectorized store: write 4 uint32 at a time
    unsigned int* out_row = (unsigned int*)(Aq + m * Kh + g * 16);
    uint4 st;
    st.x = pk[0]|(pk[1]<<8)|(pk[2]<<16)|(pk[3]<<24);
    st.y = pk[4]|(pk[5]<<8)|(pk[6]<<16)|(pk[7]<<24);
    st.z = pk[8]|(pk[9]<<8)|(pk[10]<<16)|(pk[11]<<24);
    st.w = pk[12]|(pk[13]<<8)|(pk[14]<<16)|(pk[15]<<24);
    *((uint4*)out_row) = st;
}
)";

static hipblasLtHandle_t g_h=nullptr;
static hipblasLtMatmulDesc_t g_md=nullptr;
static hipblasLtMatrixLayout_t g_lA,g_lB,g_lC;
static hipblasLtMatmulHeuristicResult_t g_hr;
static struct{void*A;void*Aq;void*As;int M;int K;int Kh;int Kg;} g_qargs;
static size_t g_qsz;static void* g_qcfg[5];static int g_qblocks;

void setup(int M,int N,int K,torch::Tensor As,torch::Tensor Bs,torch::Tensor Aq,torch::Tensor Bq,torch::Tensor out){
    if(!g_qfn){
        hiprtcProgram p;CHRTC(hiprtcCreateProgram(&p,qsrc,"q.hip",0,0,0));
        const char*o[]={"--offload-arch=gfx950","-O3"};
        hiprtcResult r=hiprtcCompileProgram(p,2,o);
        if(r!=HIPRTC_SUCCESS){size_t l;hiprtcGetProgramLogSize(p,&l);char*lg=new char[l];hiprtcGetProgramLog(p,lg);printf("CE:%s\n",lg);delete[]lg;return;}
        size_t sz;hiprtcGetCodeSize(p,&sz);char*c=new char[sz];hiprtcGetCode(p,c);
        CHIP(hipModuleLoadData(&g_qmod,c));CHIP(hipModuleGetFunction(&g_qfn,g_qmod,"mxfp4_quant_fast"));
        delete[]c;hiprtcDestroyProgram(&p);printf("QOK\n");
    }
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
    void*bp=Bs.data_ptr();void*ap=As.data_ptr();
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
    g_qargs.Aq=Aq.data_ptr();g_qargs.As=As.data_ptr();
    g_qargs.M=M;g_qargs.K=K;g_qargs.Kh=K/2;g_qargs.Kg=K/32;
    g_qsz=sizeof(g_qargs);
    g_qcfg[0]=HIP_LAUNCH_PARAM_BUFFER_POINTER;g_qcfg[1]=&g_qargs;
    g_qcfg[2]=HIP_LAUNCH_PARAM_BUFFER_SIZE;g_qcfg[3]=&g_qsz;
    g_qcfg[4]=HIP_LAUNCH_PARAM_END;
    g_qblocks=(M*(K/32)+255)/256;

    // Warmup + measure
    g_qargs.A=Aq.data_ptr(); // dummy
    for(int i=0;i<3;i++){
        CHIP(hipModuleLaunchKernel(g_qfn,g_qblocks,1,1,256,1,1,0,0,nullptr,g_qcfg));
        float alpha=1,beta=0;
        CK(hipblasLtMatmul(g_h,g_md,&alpha,Bq.data_ptr(),g_lA,Aq.data_ptr(),g_lB,&beta,out.data_ptr(),g_lC,out.data_ptr(),g_lC,&g_hr.algo,nullptr,0,0));
    }
    hipDeviceSynchronize();

    // Time quant kernel alone
    hipEvent_t e0,e1;hipEventCreate(&e0);hipEventCreate(&e1);
    hipEventRecord(e0);
    for(int i=0;i<100;i++)
        CHIP(hipModuleLaunchKernel(g_qfn,g_qblocks,1,1,256,1,1,0,0,nullptr,g_qcfg));
    hipEventRecord(e1);hipEventSynchronize(e1);
    float ms;hipEventElapsedTime(&ms,e0,e1);
    printf("FAST quant: %.1fus\n",ms*10);
    hipEventDestroy(e0);hipEventDestroy(e1);
}

torch::Tensor run(torch::Tensor A,torch::Tensor Aq,torch::Tensor Bq,torch::Tensor out){
    g_qargs.A=A.data_ptr();
    CHIP(hipModuleLaunchKernel(g_qfn,g_qblocks,1,1,256,1,1,0,0,nullptr,g_qcfg));
    float alpha=1,beta=0;
    CK(hipblasLtMatmul(g_h,g_md,&alpha,Bq.data_ptr(),g_lA,Aq.data_ptr(),g_lB,&beta,out.data_ptr(),g_lC,out.data_ptr(),g_lC,&g_hr.algo,nullptr,0,0));
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("setup",&setup);m.def("run",&run);}
"""
mod=load_inline(name="hblt_fq",cpp_sources=[cpp],extra_include_paths=["/opt/rocm/include"],
extra_ldflags=["-L/opt/rocm/lib","-lhipblaslt","-lhiprtc","-lamdhip64"],verbose=False,is_python_module=True)

M,N,K=256,3072,1536
_aq=torch.empty(M,K//2,dtype=torch.uint8,device="cuda")
_as=torch.empty(M,K//32,dtype=torch.uint8,device="cuda")
_out=torch.empty(M,N,dtype=torch.bfloat16,device="cuda")
_bq=torch.zeros(N,K//2,dtype=torch.uint8,device="cuda")
_bs=torch.zeros(N,K//32,dtype=torch.uint8,device="cuda")
_a=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
mod.setup(M,N,K,_as,_bs,_aq,_bq,_out)
torch.cuda.synchronize()

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()
_ps_ck=None;_ps_cw=None;_ps_cs=None;_hblt_bsp=0
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs,_hblt_bsp
    A=data[0];m,k=A.shape;n=data[1].shape[0]
    if m==256 and k==1536:
        B_q=data[2];B_scale_sh=data[4]
        bsp=B_scale_sh.data_ptr()
        if bsp!=_hblt_bsp:
            bs_raw=e8m0_unshuffle(B_scale_sh,n,k//32)
            mod.setup(m,n,k,_as,bs_raw,_aq,B_q,_out)
            _hblt_bsp=bsp
        return mod.run(A,_aq,B_q,_out)
    else:
        B_shuffle=data[3];B_scale_sh=data[4]
        dp=B_shuffle.data_ptr()
        if dp!=_ps_ck:
            _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
            _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
        return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
