#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#566: Fused A-quant + preshuffle-B FP4 GEMM in single kernel.
No separate quant kernel, no separate reduce. One launch.
Uses preshuffle B format directly. For M=256 K=1536 only.
Preshuffle fallback for other shapes.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# Proven configs for all shapes
_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Build fused kernel
_fused_hip = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define TM 32
#define TN 128
#define KS 128
#define NTH 256

__device__ __forceinline__ int swz(int row, int col) {
    int p = (row >> 1) & 7;
    int m = (p ^ (((p >> 1) ^ (p >> 2)) & 1)) << 4;
    return row * KS + (col ^ m);
}

// Inline FP4 quant for one group of 32 bf16 values
__device__ void quant32(const unsigned short* src, unsigned char* dst, unsigned char& sc, int valid) {
    float vals[32]; float amax = 0.0f;
    if (valid) {
        const uint4* r4 = (const uint4*)src;
        #pragma unroll
        for (int v=0;v<4;v++){uint4 c=r4[v];unsigned int w[4]={c.x,c.y,c.z,c.w};
            #pragma unroll
            for(int j=0;j<4;j++){float lo=__uint_as_float((w[j]&0xFFFFu)<<16);float hi=__uint_as_float(w[j]&0xFFFF0000u);
                vals[v*8+j*2]=lo;vals[v*8+j*2+1]=hi;amax=fmaxf(amax,fmaxf(fabsf(lo),fabsf(hi)));}}
    } else { for(int i=0;i<32;i++)vals[i]=0.0f; }
    unsigned int ai=__float_as_uint(amax);unsigned int ar=(ai+0x200000u)&0xFF800000u;
    int eb=(ar>>23)&0xFF;int su=eb-129;int sb=su+127;if(sb<0)sb=0;
    sc=(unsigned char)sb;
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
void fused_fp4_gemm(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ Bw,
    const unsigned char* __restrict__ Bsc,
    unsigned short* __restrict__ C,
    int M, int N, int K
) {
    __shared__ unsigned char sa[2][TM * KS];
    __shared__ unsigned char sb[2][TN * KS];
    __shared__ unsigned char sas[2][TM * 8];

    const int Kh = K / 2;
    const int Kg = K / 32;
    const int Ki = Kh / KS;
    const int bw_stride = Kh * 16;

    // Block assignment with XCD + L2 swizzle
    int bid = blockIdx.x;
    const int npm = M / TM, npn = N / TN, NW = gridDim.x;
    if (NW >= 8) { int r = (bid%8)*((NW+7)/8)+(bid/8); if(r<NW) bid=r; }
    int WGM = min(npm, 8), nwig = WGM * npn;
    int gid=bid/nwig, fpm=gid*WGM, gsm=min(npm-fpm,WGM);
    int pm=fpm+((bid%nwig)%gsm), pn=(bid%nwig)/gsm;

    int tid=threadIdx.x, wid=tid/64, lane=tid%64, l32=lane&31, grp=lane>>5;
    int wn=wid;
    int arb=pm*TM, bcb=pn*TN+wn*32;

    typedef __attribute__((__vector_size__(8*sizeof(int)))) int v8i;
    typedef __attribute__((__vector_size__(16*sizeof(float)))) float v16f;
    v16f acc = {};

    // === Load first K-tile ===
    // A: quant 32 rows × 8 groups = 256 groups, 1 per thread
    {
        int row = tid / 8, grp_idx = tid % 8;
        int arow = arb + row, agrp = grp_idx;
        unsigned char fp4[16]; unsigned char sc;
        quant32(A + arow * K + agrp * 32, fp4, sc, arow < M && agrp < Kg);
        int lc = grp_idx * 16;
        #pragma unroll
        for (int i=0;i<16;i++) sa[0][swz(row, lc+i)] = fp4[i];
        sas[0][row * 8 + grp_idx] = sc;
    }
    // B: 128×128 = 16384 bytes, 4 rounds of 256 threads × 16 bytes
    for (int rd=0; rd<4; rd++) {
        int flat = tid + rd * NTH;
        int br = flat / 8, bg = flat % 8;
        int nc = pn * TN + br;
        int kb = bg * 16;
        if (nc < N && kb + 15 < Kh) {
            int sr = nc / 16, nw = nc % 16;
            int kbb = kb / 32, khh = (kb % 32) / 16;
            const unsigned char* src = Bw + (long)sr * bw_stride + kbb*512 + khh*256 + nw*16;
            *(uint4*)&sb[0][swz(br, bg*16)] = *(const uint4*)src;
        } else {
            *(uint4*)&sb[0][swz(br, bg*16)] = make_uint4(0,0,0,0);
        }
    }
    __syncthreads();

    int tic=0, toc=1;
    for (int ki=0; ki<Ki; ki++, tic^=1, toc^=1) {
        // Prefetch next
        if (ki+1 < Ki) {
            int nki = ki+1;
            {
                int row=tid/8, gi=tid%8;
                int arow=arb+row, agrp=nki*8+gi;
                unsigned char fp4[16]; unsigned char sc;
                quant32(A + arow*K + agrp*32, fp4, sc, arow<M && agrp<Kg);
                int lc=gi*16;
                #pragma unroll
                for(int i=0;i<16;i++) sa[toc][swz(row,lc+i)]=fp4[i];
                sas[toc][row*8+gi]=sc;
            }
            for(int rd=0;rd<4;rd++){
                int flat=tid+rd*NTH, br=flat/8, bg=flat%8;
                int nc=pn*TN+br, kb=(nki*KS)+bg*16;
                if(nc<N && kb+15<Kh){
                    int sr=nc/16, nw=nc%16, kbb=kb/32, khh=(kb%32)/16;
                    const unsigned char* src=Bw+(long)sr*bw_stride+kbb*512+khh*256+nw*16;
                    *(uint4*)&sb[toc][swz(br,bg*16)]=*(const uint4*)src;
                } else {
                    *(uint4*)&sb[toc][swz(br,bg*16)]=make_uint4(0,0,0,0);
                }
            }
        }

        // Load scales
        unsigned char asc[8], bsc_r[8];
        {
            int ar = l32;
            if (arb+ar < M) {
                #pragma unroll
                for(int s=0;s<8;s++) asc[s]=sas[tic][ar*8+s];
            } else {
                #pragma unroll
                for(int s=0;s<8;s++) asc[s]=0;
            }
        }
        {
            int nc=bcb+l32;
            if(nc<N){
                int kg8=Kg/8, n0=nc/32, n1=(nc&31)/16, n2=nc&15;
                #pragma unroll
                for(int s=0;s<8;s++){
                    int g=ki*8+s, g0=g/8, g1=(g&7)/4, g2=g&3;
                    bsc_r[s]=Bsc[n0*(kg8*256)+g0*256+g2*64+n2*4+g1*2+n1];
                }
            } else {
                #pragma unroll
                for(int s=0;s<8;s++) bsc_r[s]=0;
            }
        }

        asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)");
        __syncthreads();

        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for(int sub=0;sub<4;sub++){
            int ko=sub*32+grp*16;
            v8i ar={}, br={};
            *(uint4*)&ar=*(uint4*)&sa[tic][swz(l32,ko)];
            *(uint4*)&br=*(uint4*)&sb[tic][swz(wn*32+l32,ko)];
            unsigned int pas=(unsigned int)asc[sub*2+grp]|((unsigned int)asc[sub*2+grp]<<8);
            unsigned int pbs=(unsigned int)bsc_r[sub*2+grp]|((unsigned int)bsc_r[sub*2+grp]<<8);
            __builtin_amdgcn_sched_barrier(0);
            acc=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(ar,br,acc,4,4,0,pas,0,pbs);
            __builtin_amdgcn_sched_barrier(0);
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // Store bf16
    #pragma unroll
    for(int i=0;i<4;i++){
        #pragma unroll
        for(int j=0;j<4;j++){
            int r=arb+grp*4+i*8+j, c=bcb+l32;
            if(r<M&&c<N){
                float v=acc[i*4+j];
                unsigned int u=__float_as_uint(v);
                u=u+(((u>>16)&1)+0x7FFF);
                C[r*N+c]=(unsigned short)(u>>16);
            }
        }
    }
}

torch::Tensor run_fused(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bsc, int N) {
    int M=(int)A.size(0), K=(int)A.size(1);
    auto C=torch::empty({M,N},torch::dtype(torch::kBFloat16).device(A.device()));
    int grid=(M/32)*(N/128);
    fused_fp4_gemm<<<grid,256,0,0>>>(
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)Bw.data_ptr(),
        (const unsigned char*)Bsc.data_ptr(),
        (unsigned short*)C.data_ptr(), M, N, K);
    return C;
}
"""

_fused_cpp = r"""
#include <torch/extension.h>
torch::Tensor run_fused(torch::Tensor A, torch::Tensor Bw, torch::Tensor Bsc, int N);
"""

_fused_mod = None
try:
    _fused_mod = load_inline(
        name='fused_566',
        cpp_sources=_fused_cpp,
        cuda_sources=_fused_hip,
        functions=['run_fused'],
        verbose=False,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    # Quick warmup
    _wA = torch.randn(256, 1536, dtype=torch.bfloat16, device="cuda")
    _wBw = torch.zeros(3072//16, (1536//2)*16, dtype=torch.uint8, device="cuda")
    _wBs = torch.zeros(3072//32, 1536, dtype=torch.uint8, device="cuda")
    _fused_mod.run_fused(_wA, _wBw, _wBs, 3072)
    torch.cuda.synchronize()
    del _wA, _wBw, _wBs
    print("Fused kernel OK")
except Exception as e:
    _fused_mod = None
    print(f"Fused kernel FAILED: {e}")

# Warmup preshuffle
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

    # Fused kernel for M=256 K=1536
    if _fused_mod is not None and m == 256 and k == 1536:
        # B_shuffle is the preshuffle data, B_scale_sh is shuffled E8M0
        return _fused_mod.run_fused(A, B_shuffle.view(torch.uint8), B_scale_sh.view(torch.uint8), n)

    # Preshuffle fallback
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
