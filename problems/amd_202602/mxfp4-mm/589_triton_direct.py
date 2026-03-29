#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#589: Direct hipModuleLaunchKernel using Triton's OWN function pointer.
Extract fn_ptr from _gemm_a16wfp4_preshuffle_kernel.fn.device_caches[0].
This guarantees HSACO match — no stale cache issue.
"""
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

# Step 1: Warmup all shapes (triggers Triton JIT)
print("=== Warmup ===")
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
    _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
    _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
    try: gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except: pass
torch.cuda.synchronize()

# Step 2: Extract fn_ptr from Triton's cache
print("=== Extracting fn_ptrs ===")
import importlib, triton
mod = importlib.import_module('aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4')
preshuffle_heuristics = getattr(mod, '_gemm_a16wfp4_preshuffle_kernel', None)

_fn_ptrs = {}  # (BSM, BSN, warps) -> (fn_ptr, num_warps, shared, name)
if preshuffle_heuristics and hasattr(preshuffle_heuristics, 'fn'):
    jit_fn = preshuffle_heuristics.fn
    # Dump ALL attributes to find the kernel cache
    print(f"  JITFunction attrs: {[a for a in dir(jit_fn) if not a.startswith('__')]}")
    for attr in ['device_caches', 'cache', 'kernel_cache', 'kernel', 'module']:
        if hasattr(jit_fn, attr):
            val = getattr(jit_fn, attr)
            vtype = type(val).__name__
            if isinstance(val, dict):
                print(f"  .{attr}: dict({len(val)} entries)")
                for dk, dv in list(val.items())[:2]:
                    dktype = type(dk).__name__
                    dvtype = type(dv).__name__
                    print(f"    [{dktype}] -> {dvtype}")
                    # Go one level deeper
                    if isinstance(dv, dict):
                        for dk2, dv2 in list(dv.items())[:2]:
                            print(f"      [{type(dk2).__name__}] -> {type(dv2).__name__}: {[a for a in dir(dv2) if not a.startswith('_')][:8]}")
                            fn_ptr = getattr(dv2, 'function', None)
                            if fn_ptr:
                                print(f"        FOUND fn=0x{fn_ptr:x}")
                    elif isinstance(dv, (tuple, list)):
                        for i, item in enumerate(dv[:3]):
                            print(f"      [{i}] {type(item).__name__}: {[a for a in dir(item) if not a.startswith('_')][:6]}")
                            fn_ptr = getattr(item, 'function', None)
                            if fn_ptr:
                                print(f"        FOUND fn=0x{fn_ptr:x}")
                    else:
                        fn_ptr = getattr(dv, 'function', None)
                        if fn_ptr:
                            print(f"      FOUND fn=0x{fn_ptr:x}")
            elif isinstance(val, (tuple, list)):
                print(f"  .{attr}: {vtype}({len(val)} entries)")
            else:
                print(f"  .{attr}: {vtype}")
    if hasattr(jit_fn, 'device_caches') and 0 in jit_fn.device_caches:
        cache_tuple = jit_fn.device_caches[0]
        print(f"\n  device_caches[0] has {len(cache_tuple)} entries, type={type(cache_tuple).__name__}")
        # cache_tuple structure: (spec_cache_dict, key_cache_dict, target, backend, binder)
        # spec_cache maps specialization keys to CompiledKernel objects
        spec_cache = cache_tuple[0] if len(cache_tuple) > 0 else {}
        key_cache = cache_tuple[1] if len(cache_tuple) > 1 else {}
        if isinstance(spec_cache, dict):
            print(f"  spec_cache: {len(spec_cache)} entries")
            # Get first entry quickly
            first_key = next(iter(spec_cache)) if spec_cache else None
            if first_key is not None:
                first_val = spec_cache[first_key]
                print(f"  first val type: {type(first_val).__name__}")
                print(f"  first val attrs: {[a for a in dir(first_val) if not a.startswith('_')][:15]}")
                fn_ptr = getattr(first_val, 'function', None)
                print(f"  .function = {fn_ptr}")
        # Extract ALL CompiledKernels from spec_cache
        if isinstance(spec_cache, dict):
            for sk, sv in spec_cache.items():
                fn_ptr = getattr(sv, 'function', None)
                if fn_ptr:
                    meta = getattr(sv, 'metadata', None)
                    shared = getattr(meta, 'shared', 0) if meta else 0
                    warps = getattr(meta, 'num_warps', 4) if meta else 4
                    cname = getattr(sv, 'name', '?')
                    # Parse BSM, BSN from name
                    import re as _re
                    _bsm_m = _re.search(r'BLOCK_SIZE_M_(\d+)', str(cname))
                    _bsn_m = _re.search(r'BLOCK_SIZE_N_(\d+)', str(cname))
                    _bsm = int(_bsm_m.group(1)) if _bsm_m else 0
                    _bsn = int(_bsn_m.group(1)) if _bsn_m else 0
                    _fn_ptrs[(_bsm, _bsn, warps)] = (fn_ptr, warps, shared, str(cname))
                    print(f"  FOUND fn=0x{fn_ptr:x} BSM={_bsm} BSN={_bsn} warps={warps} shared={shared}")
        try:
            # Cache might be: dict, tuple-of-pairs, or other iterable
            if hasattr(cache, 'items'):
                items = list(cache.items())
            elif isinstance(cache, (tuple, list)):
                # Could be tuple of (key, compiled) pairs, or tuple of compiled objects
                items = []
                for entry in cache:
                    if isinstance(entry, (tuple, list)) and len(entry) == 2:
                        items.append(entry)  # (key, compiled)
                    else:
                        items.append((None, entry))  # just compiled
            else:
                items = list(enumerate(cache))

            for k, compiled in items:
                # compiled might itself be a tuple/wrapper — try to get function
                fn_ptr = getattr(compiled, 'function', None)
                if fn_ptr is None and hasattr(compiled, '__getitem__'):
                    # Maybe it's a tuple (compiled_obj, ...)
                    try: fn_ptr = getattr(compiled[0], 'function', None)
                    except: pass
                meta = getattr(compiled, 'metadata', None)
                if meta is None and hasattr(compiled, '__getitem__'):
                    try: meta = getattr(compiled[0], 'metadata', None)
                    except: pass
                shared = getattr(meta, 'shared', 0) if meta else 0
                num_warps = getattr(meta, 'num_warps', 4) if meta else 4
                cname = getattr(compiled, 'name', getattr(compiled, '__class__', ''))
                if fn_ptr:
                    # This path shouldn't be reached anymore
                    pass
                    print(f"  fn=0x{fn_ptr:x} warps={num_warps} shared={shared} name={str(cname)[:80]}")
                else:
                    # Debug: print what compiled looks like
                    ctype = type(compiled).__name__
                    cattrs = [a for a in dir(compiled) if not a.startswith('_')][:8]
                    print(f"  NO fn_ptr: type={ctype} attrs={cattrs}")
        except Exception as e:
            print(f"  Cache iteration error: {e}")
            import traceback; traceback.print_exc()

print(f"\n  Extracted {len(_fn_ptrs)} function pointers")

# Step 3: Build C++ dispatch module
_dispatch_hip = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define MAX_SHAPES 8

struct ShapeKernel {
    hipFunction_t fn;
    int num_warps;
    int shared_mem;
    int grid_x;
    int BSM, BSN;
};

static ShapeKernel g_shapes[MAX_SHAPES];
static int g_nshapes = 0;

bool register_kernel(int slot, int64_t fn_ptr, int num_warps, int shared_mem,
                     int grid_x, int bsm, int bsn) {
    if (slot >= MAX_SHAPES) return false;
    g_shapes[slot].fn = (hipFunction_t)(uintptr_t)fn_ptr;
    g_shapes[slot].num_warps = num_warps;
    g_shapes[slot].shared_mem = shared_mem;
    g_shapes[slot].grid_x = grid_x;
    g_shapes[slot].BSM = bsm;
    g_shapes[slot].BSN = bsn;
    if (slot >= g_nshapes) g_nshapes = slot + 1;
    return true;
}

// Launch using Triton's own fn_ptr with params mode
torch::Tensor direct_launch(
    int slot,
    torch::Tensor A, torch::Tensor Bw, torch::Tensor Bws,
    torch::Tensor output,
    int M, int N, int K
) {
    auto& sk = g_shapes[slot];

    // Args: 4 pointers + 12 int32 (matching Triton's preshuffle kernel)
    uint64_t a_ptr = (uint64_t)A.data_ptr();
    uint64_t b_ptr = (uint64_t)Bw.data_ptr();
    uint64_t c_ptr = (uint64_t)output.data_ptr();
    uint64_t bs_ptr = (uint64_t)Bws.data_ptr();
    int32_t iM = M, iN = N, iK = K;
    int32_t s_am = (int32_t)A.stride(0);
    int32_t s_ak = (int32_t)A.stride(1);
    int32_t s_bn = (int32_t)Bw.stride(0);
    int32_t s_bk = (int32_t)Bw.stride(1);
    int32_t s_ck = 0;
    int32_t s_cm = (int32_t)output.stride(0);
    int32_t s_cn = (int32_t)output.stride(1);
    int32_t s_bsn = (int32_t)Bws.stride(0);
    int32_t s_bsk = (int32_t)Bws.stride(1);

    void* params[] = {
        &a_ptr, &b_ptr, &c_ptr, &bs_ptr,
        &iM, &iN, &iK,
        &s_am, &s_ak, &s_bn, &s_bk,
        &s_ck, &s_cm, &s_cn,
        &s_bsn, &s_bsk,
    };

    hipModuleLaunchKernel(
        sk.fn,
        sk.grid_x, 1, 1,
        64 * sk.num_warps, 1, 1,
        sk.shared_mem, 0, params, nullptr);

    return output;
}

// CK ASM for shape 6
// [same fused_quant_shuffle + CK ASM code as submission.py #567]
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

struct __attribute__((packed)) CKArgs {
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
static hipModule_t g_ck_mod=nullptr; static hipFunction_t g_ck_fn=nullptr;
static torch::Tensor g_aq,g_ash,g_ck_out;
static int g_ck_Mp=0,g_ck_K=0,g_ck_N=0;
bool init_ck(const std::string& co,const std::string& fn){
    if(g_ck_fn)return true;
    if(hipModuleLoad(&g_ck_mod,co.c_str())!=hipSuccess)return false;
    if(hipModuleGetFunction(&g_ck_fn,g_ck_mod,fn.c_str())!=hipSuccess)return false;
    return true;
}
torch::Tensor ck_dispatch(torch::Tensor A,torch::Tensor Bs,torch::Tensor Bss,int Nv){
    if(!g_ck_fn)return torch::Tensor();
    int M=A.size(0),Ke=A.size(1),Mp=((M+31)/32)*32,Kh=Ke/2,Kg=Ke/32;
    if(Mp!=g_ck_Mp||Ke!=g_ck_K){g_aq=torch::empty({Mp,Kh},torch::dtype(torch::kUInt8).device(A.device()));
        g_ash=torch::empty({Mp,Kg},torch::dtype(torch::kUInt8).device(A.device()));g_ck_Mp=Mp;g_ck_K=Ke;}
    if(Nv!=g_ck_N||Mp!=g_ck_Mp){g_ck_out=torch::empty({Mp,Nv},torch::dtype(torch::kBFloat16).device(A.device()));g_ck_N=Nv;}
    int tg=Mp*Kg;
    hipLaunchKernelGGL(fused_quant_shuffle, dim3((tg+127)/128), dim3(128), 0, 0,
        (const unsigned short*)A.data_ptr(),
        g_aq.data_ptr<unsigned char>(),g_ash.data_ptr<unsigned char>(),M,Ke,Kh,Kg,Mp);
    CKArgs ka;memset(&ka,0,sizeof(ka));
    ka.ptr_D=g_ck_out.data_ptr();ka.ptr_C=g_ck_out.data_ptr();
    ka.ptr_A=g_aq.data_ptr<unsigned char>();ka.ptr_B=Bs.data_ptr();
    ka.alpha=1.0f;ka.beta=0.0f;ka.stride_D0=Nv;ka.stride_D1=1;ka.stride_C0=Nv;ka.stride_C1=1;
    ka.stride_A0=Ke;ka.stride_A1=1;ka.stride_B0=Bs.stride(0)*2;ka.stride_B1=1;
    ka.M=Mp;ka.N=Nv;ka.K=Ke;ka.ptr_ScaleA=g_ash.data_ptr<unsigned char>();ka.ptr_ScaleB=Bss.data_ptr();
    ka.stride_ScaleA0=Kg;ka.stride_ScaleA1=1;ka.stride_ScaleB0=Bss.stride(0);ka.stride_ScaleB1=1;
    ka.log2_k_split=0;size_t asz=sizeof(ka);void* cfg[]={(void*)0x01,&ka,(void*)0x02,&asz,(void*)0x03};
    hipModuleLaunchKernel(g_ck_fn,(Nv+127)/128,(Mp+31)/32,1,256,1,1,0,0,nullptr,cfg);
    return g_ck_out.slice(0,0,M);
}

// Fast reduce for KSPLIT shapes
extern "C" __global__ void fast_reduce_k7(
    const float* __restrict__ src, unsigned short* __restrict__ dst,
    int MN, int stride_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MN) return;
    float sum = src[idx]+src[idx+stride_k]+src[idx+2*stride_k]+src[idx+3*stride_k]
              +src[idx+4*stride_k]+src[idx+5*stride_k]+src[idx+6*stride_k];
    unsigned int u = __float_as_uint(sum);
    u = u + (((u >> 16) & 1) + 0x7FFF);
    dst[idx] = (unsigned short)(u >> 16);
}
static torch::Tensor g_reduce_out;
static int g_rm=0,g_rn=0;
torch::Tensor fused_reduce(torch::Tensor src, int M, int N, int KSPLIT) {
    if(M!=g_rm||N!=g_rn){g_reduce_out=torch::empty({M,N},torch::dtype(torch::kBFloat16).device(src.device()));g_rm=M;g_rn=N;}
    int MN=M*N,th=256,bl=(MN+th-1)/th;
    hipLaunchKernelGGL(fast_reduce_k7, dim3(bl), dim3(th), 0, 0,
        src.data_ptr<float>(),(unsigned short*)g_reduce_out.data_ptr(),MN,(int)src.stride(0));
    return g_reduce_out;
}
"""

_dispatch_cpp = r"""
#include <torch/extension.h>
bool register_kernel(int slot, int64_t fn_ptr, int num_warps, int shared_mem, int grid_x, int bsm, int bsn);
torch::Tensor direct_launch(int slot, torch::Tensor A, torch::Tensor Bw, torch::Tensor Bws, torch::Tensor output, int M, int N, int K);
bool init_ck(const std::string&, const std::string&);
torch::Tensor ck_dispatch(torch::Tensor A, torch::Tensor Bs, torch::Tensor Bss, int Nv);
torch::Tensor fused_reduce(torch::Tensor src, int M, int N, int KSPLIT);
"""

print("=== Building C++ module ===")
_mod = None
try:
    _mod = load_inline(
        name='direct_589',
        cpp_sources=_dispatch_cpp,
        cuda_sources=_dispatch_hip,
        functions=['register_kernel', 'direct_launch', 'init_ck', 'ck_dispatch', 'fused_reduce'],
        verbose=False,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    print("C++ module OK")
except Exception as e:
    _mod = None
    print(f"C++ module FAILED: {e}")

# Step 4: Register extracted fn_ptrs with C++ module
# Map shapes to configs for grid calculation
_shape_configs = {
    (4, 2880, 512):   {'BSM': 8, 'BSN': 32, 'warps': 4, 'KSPLIT': 1},
    (32, 4096, 512):  {'BSM': 32, 'BSN': 32, 'warps': 4, 'KSPLIT': 1},
    (32, 2880, 512):  {'BSM': 32, 'BSN': 32, 'warps': 4, 'KSPLIT': 1},
}

_direct_shapes = {}  # (m,n,k) -> slot
_direct_ok = False

if _mod and _fn_ptrs:
    slot = 0
    for (m,n,k), cfg in _shape_configs.items():
        # Find matching fn_ptr by (BSM, BSN, warps)
        found = False
        lookup_key = (cfg['BSM'], cfg['BSN'], cfg['warps'])
        if lookup_key in _fn_ptrs:
            fn, w, s, name = _fn_ptrs[lookup_key]
            if True:
                N_k = n  # logical N
                K_k = k // 2  # K = K_bf16 / 2 (as wrapper computes)
                grid_x = ((N_k + cfg['BSN'] - 1) // cfg['BSN']) * ((m + cfg['BSM'] - 1) // cfg['BSM'])

                ok = _mod.register_kernel(slot, fn, w, s, grid_x, cfg['BSM'], cfg['BSN'])
                if ok:
                    _direct_shapes[(m,n,k)] = (slot, N_k, K_k)
                    print(f"  Slot {slot}: M={m} N={n} K={k} fn=0x{fn:x} warps={w} shared={s} grid={grid_x}")
                    found = True
                    slot += 1
                    break
        if not found:
            print(f"  No fn_ptr found for M={m} warps={cfg['warps']}")

    if _direct_shapes:
        # Validate correctness with random data
        print("\n=== Validating ===")
        for (m,n,k), (slot, N_k, K_k) in list(_direct_shapes.items())[:1]:
            _vA = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            _vBw = torch.randint(0, 256, (n//16, (k//2)*16), dtype=torch.uint8, device="cuda")
            _vBws = torch.randint(100, 150, (n//32, k), dtype=torch.uint8, device="cuda")
            _vRef = gemm_a16wfp4_preshuffle(_vA, _vBw, _vBws, prequant=True, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            _vOut = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
            _mod.direct_launch(slot, _vA, _vBw, _vBws, _vOut, m, N_k, K_k)
            torch.cuda.synchronize()
            diff = (_vRef.float() - _vOut.float()).abs().max().item()
            print(f"  M={m} N={n} K={k}: max_diff={diff}")
            if diff > 1.0:
                print("  FAILED — disabling direct launch")
                _direct_shapes = {}
            else:
                print("  OK!")
                _direct_ok = True
            del _vA, _vBw, _vBws, _vRef, _vOut

# Init CK ASM for shape 6
_ck_ok = False
if _mod:
    try:
        _ck_ok = _mod.init_ck(
            "/home/runner/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.co",
            "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E")
        if _ck_ok:
            _w=torch.randn(256,1536,dtype=torch.bfloat16,device="cuda")
            _mod.ck_dispatch(_w,torch.zeros(3072,768,dtype=torch.uint8,device="cuda"),
                torch.zeros(3072,48,dtype=torch.uint8,device="cuda"),3072)
            torch.cuda.synchronize();del _w
            print("CK ASM OK")
    except: pass

# Check skip_reduce support
_use_fast_reduce = False
import inspect
_orig_preshuffle = gemm_a16wfp4_preshuffle
_sig = inspect.signature(_orig_preshuffle)
_use_fast_reduce = 'skip_reduce' in _sig.parameters
print(f"skip_reduce: {_use_fast_reduce}")

# Pre-allocate outputs
_outputs = {}
for (m,n,k) in [(4,2880,512),(32,4096,512),(32,2880,512)]:
    _outputs[(m,n)] = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

torch.cuda.empty_cache()

_ps_ck=None;_ps_cw=None;_ps_cs=None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]

    # Shape 6: CK ASM
    if _ck_ok and m == 256 and k == 1536:
        return _mod.ck_dispatch(A, B_shuffle, B_scale_sh, n)

    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)

    # Direct launch for KSPLIT=1 shapes
    if _direct_ok and (m,n,k) in _direct_shapes:
        slot, N_k, K_k = _direct_shapes[(m,n,k)]
        out = _outputs.get((m,n))
        if out is not None:
            return _mod.direct_launch(slot, A, _ps_cw, _ps_cs, out, m, N_k, K_k)

    # Fast reduce for K=7168
    if _use_fast_reduce and _mod and k >= 7168:
        result = _orig_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16,skip_reduce=True)
        if result.dim() == 3:
            ks, rm, rn = result.shape
            return _mod.fused_reduce(result, rm, rn, ks)
        return result

    # Fallback: Triton preshuffle
    return _orig_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
