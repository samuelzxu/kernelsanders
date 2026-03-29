#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#593: Hand-tuned fused quant+GEMM v2. Fuses A quant INTO K-loop.
Double-buffered LDS, per-K-step inline quant, sched_barrier, s_setprio.
Tests shapes 3/4 (M=32 K=512). Other shapes use proven paths.
"""
import torch, os, json, inspect
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

# Read the kernel source from the .hip file (it was compiled locally,
# but we JIT-compile on runner via load_inline for compatibility)
_fused_src = r"""
""" + open('/dev/null').read()  # placeholder — inline the source below

# Inline the complete kernel source
_fused_src = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#define TM 32
#define TN 32
#define K_STEP 128
#define NW 4
#define NTH (64*NW)
__device__ __forceinline__ int swz_b(int row, int col) {
    int p=(row>>1)&7; int m=(p^(((p>>1)^(p>>2))&1))<<4;
    return row*K_STEP+(col^m);
}
__device__ __forceinline__ void quant_group(const unsigned short* src, unsigned char* dst, unsigned char& sc) {
    float vals[32]; float amax=0.0f;
    const uint4* r4=(const uint4*)src;
    #pragma unroll
    for(int v=0;v<4;v++){uint4 c=r4[v];unsigned int w[4]={c.x,c.y,c.z,c.w};
        #pragma unroll
        for(int j=0;j<4;j++){float lo=__uint_as_float((w[j]&0xFFFFu)<<16);float hi=__uint_as_float(w[j]&0xFFFF0000u);
            vals[v*8+j*2]=lo;vals[v*8+j*2+1]=hi;amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi)));}}
    unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
    int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;sc=(unsigned char)sb;
    int qe=127-su;if(qe<1)qe=0;if(qe>254)qe=254;
    float qs=__uint_as_float((unsigned int)qe<<23);if(amax==0.0f)qs=0.0f;
    const unsigned int dmi=149u<<23;float dmf=__uint_as_float(dmi);
    const int vta=((int)(1-127)<<23)+(1<<21)-1;
    auto fp4=[&](float v)->unsigned char{float qf=v*qs;unsigned int qx=__float_as_uint(qf);
        unsigned int s2=qx&0x80000000u;qx^=s2;float qp=__uint_as_float(qx);unsigned char r;
        if(qp>=6.0f)r=0x7;else if(qp<1.0f)r=(unsigned char)((__float_as_uint(qp+dmf)-dmi)&0xFF);
        else{unsigned int mo=(qx>>22)&1;r=(unsigned char)((((unsigned int)((int)qx+vta)+mo)>>22)&0xFF);}
        return(r&0x7)|((unsigned char)(s2>>28)&0x8);};
    #pragma unroll
    for(int i=0;i<16;i++)dst[i]=fp4(vals[2*i])|(fp4(vals[2*i+1])<<4);
}
extern "C" __global__ __launch_bounds__(NTH, 2)
void fp4_fused_v2(const unsigned short* __restrict__ A, const unsigned char* __restrict__ Bw,
    const unsigned char* __restrict__ Bs, unsigned short* __restrict__ C, int M, int N, int K) {
    const int Kh=K/2,Kg=K/32,bw_stride=Kh*16,K_steps=Kh/K_STEP;
    int bid=blockIdx.x;
    const int npm=(M+TM-1)/TM,npn=(N+TN-1)/TN,NWg=gridDim.x;
    if(NWg>=8){int r=(bid%8)*((NWg+7)/8)+(bid/8);if(r<NWg)bid=r;}
    int WGM=min(npm,8),nwig=WGM*npn,gid=bid/nwig,fpm=gid*WGM,gsm=min(npm-fpm,WGM);
    int pm=fpm+((bid%nwig)%gsm),pn=(bid%nwig)/gsm;
    int tid=threadIdx.x,lane=tid%64,l32=lane&31,grp=lane>>5;
    int a_row=pm*TM,b_col=pn*TN;
    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc={};
    __shared__ unsigned char lds_aq[2][TM*K_STEP];
    __shared__ unsigned char lds_b[2][TN*K_STEP];
    __shared__ unsigned char lds_as[2][TM*8];
    int tic=0,toc=1;
    // PROLOGUE: load first K-step
    {
        int gi=tid;
        if(gi<TM*(K_STEP/16)){
            int row=gi/(K_STEP/16),g=gi%(K_STEP/16);
            int ar2=a_row+row,ag=g;
            unsigned char fp4[16];unsigned char sc;
            if(ar2<M&&ag<Kg)quant_group(A+ar2*K+ag*32,fp4,sc);
            else{for(int i=0;i<16;i++)fp4[i]=0;sc=0;}
            *(uint4*)&lds_aq[0][row*K_STEP+g*16]=*(uint4*)fp4;
            lds_as[0][row*8+g]=sc;
        }
        {
            int flat=tid*16,br=flat/K_STEP,bc=flat%K_STEP;
            int nc=b_col+br,kb=bc;
            if(nc<N&&kb<Kh){
                int sr=nc/16,nw=nc%16,kbb=kb/32,khh=(kb%32)/16;
                const unsigned char*src=Bw+(long)sr*bw_stride+kbb*512+khh*256+nw*16;
                *(uint4*)&lds_b[0][swz_b(br,bc)]=*(const uint4*)src;
            }else{*(uint4*)&lds_b[0][swz_b(br,bc)]=make_uint4(0,0,0,0);}
        }
    }
    __syncthreads();
    // K-LOOP
    for(int ki=0;ki<K_steps;ki++,tic^=1,toc^=1){
        if(ki+1<K_steps){
            int nk=(ki+1)*K_STEP;
            int gi=tid;
            if(gi<TM*(K_STEP/16)){
                int row=gi/(K_STEP/16),g=gi%(K_STEP/16);
                int ar2=a_row+row,ag=(ki+1)*(K_STEP/16)+g;
                unsigned char fp4[16];unsigned char sc;
                if(ar2<M&&ag<Kg)quant_group(A+ar2*K+ag*32,fp4,sc);
                else{for(int i=0;i<16;i++)fp4[i]=0;sc=0;}
                *(uint4*)&lds_aq[toc][row*K_STEP+g*16]=*(uint4*)fp4;
                lds_as[toc][row*8+g]=sc;
            }
            {
                int flat=tid*16,br=flat/K_STEP,bc=flat%K_STEP;
                int nc=b_col+br,kb=nk+bc;
                if(nc<N&&kb<Kh){
                    int sr=nc/16,nw=nc%16,kbb=kb/32,khh=(kb%32)/16;
                    const unsigned char*src=Bw+(long)sr*bw_stride+kbb*512+khh*256+nw*16;
                    *(uint4*)&lds_b[toc][swz_b(br,bc)]=*(const uint4*)src;
                }else{*(uint4*)&lds_b[toc][swz_b(br,bc)]=make_uint4(0,0,0,0);}
            }
        }
        unsigned char bsc[8];
        {
            int nc=b_col+l32,sb2=ki*(K_STEP/16);
            if(nc<N){int kg8=Kg/8,n0=nc/32,n1=(nc&31)/16,n2=nc&15;
                #pragma unroll
                for(int s=0;s<8;s++){int g=sb2+s,g0=g/8,g1=(g&7)/4,g2=g&3;
                    bsc[s]=Bs[n0*(kg8*256)+g0*256+g2*64+n2*4+g1*2+n1];}
            }else{for(int s=0;s<8;s++)bsc[s]=0;}
        }
        unsigned char asc[8];
        {
            int row=l32;
            #pragma unroll
            for(int s=0;s<8;s++)asc[s]=(a_row+row<M)?lds_as[tic][row*8+s]:(unsigned char)0;
        }
        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for(int sub=0;sub<4;sub++){
            int ko=sub*32+grp*16;
            v8i ar={},br={};
            *(uint4*)&ar=*(uint4*)&lds_aq[tic][l32*K_STEP+ko];
            *(uint4*)&br=*(uint4*)&lds_b[tic][swz_b(l32,ko)];
            unsigned int pas=(unsigned int)asc[sub*2+grp]|((unsigned int)asc[sub*2+grp]<<8);
            unsigned int pbs=(unsigned int)bsc[sub*2+grp]|((unsigned int)bsc[sub*2+grp]<<8);
            __builtin_amdgcn_sched_barrier(0);
            acc=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(ar,br,acc,4,4,0,pas,0,pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            int r=a_row+grp*4+i*8+j,c=b_col+l32;
            if(r<M&&c<N){float v=acc[i*4+j];unsigned int u=__float_as_uint(v);
                u=u+(((u>>16)&1)+0x7FFF);C[r*N+c]=(unsigned short)(u>>16);}
        }
    }
}
static torch::Tensor g_out;static int g_m=0,g_n=0;
torch::Tensor run_fused_v2(torch::Tensor A,torch::Tensor Bw,torch::Tensor Bs,int N_val){
    int M=(int)A.size(0),K=(int)A.size(1);
    if(M!=g_m||N_val!=g_n){g_out=torch::empty({M,N_val},torch::dtype(torch::kBFloat16).device(A.device()));g_m=M;g_n=N_val;}
    int grid=((M+TM-1)/TM)*((N_val+TN-1)/TN);
    hipLaunchKernelGGL(fp4_fused_v2,dim3(grid),dim3(NTH),0,0,
        (const unsigned short*)A.data_ptr(),(const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bs.data_ptr(),(unsigned short*)g_out.data_ptr(),M,N_val,K);
    return g_out;
}
"""
_fused_cpp = r"""
#include <torch/extension.h>
torch::Tensor run_fused_v2(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bs, int N_val);
"""
_fused_mod = None
try:
    _fused_mod = load_inline(name='fv2_593',cpp_sources=_fused_cpp,cuda_sources=_fused_src,
        functions=['run_fused_v2'],verbose=False,
        extra_cuda_cflags=['-O3','-w','-mcumode','--offload-arch=gfx950'])
    _w=torch.randn(32,512,dtype=torch.bfloat16,device="cuda")
    _fused_mod.run_fused_v2(_w,torch.zeros(256,4096,dtype=torch.uint8,device="cuda"),
        torch.zeros(128,512,dtype=torch.uint8,device="cuda"),4096)
    torch.cuda.synchronize();del _w
    print("Fused v2 OK")
except Exception as e:
    print(f"Fused v2 FAILED: {e}")

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

    # Fused v2 for K=512 M=32 shapes
    if _fused_mod is not None and m == 32 and k == 512:
        return _fused_mod.run_fused_v2(A, _ps_cw, _ps_cs, n)

    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
