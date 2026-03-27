#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#341: Compare hiprtc quant vs dynamic_mxfp4_quant for M=256 K=1536.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
from aiter.ops.triton.quant import dynamic_mxfp4_quant

hiprtc_quant_code = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#define CHRTC(x) do { hiprtcResult r=(x); if(r!=HIPRTC_SUCCESS) printf("RTC E%d@%d: %s\n",(int)r,__LINE__,hiprtcGetErrorString(r)); } while(0)
#define CHIP(x) do { hipError_t e=(x); if(e) printf("HIP E%d@%d\n",(int)e,__LINE__); } while(0)
static hipModule_t g_mod=nullptr; static hipFunction_t g_fn=nullptr;
static const char* src = R"(
extern "C" __global__ void mxfp4_quant(
    const unsigned short* __restrict__ A,
    unsigned char* __restrict__ A_q,
    unsigned char* __restrict__ A_scale,
    int M, int K, int K_half, int K_groups
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= M * K_groups) return;
    int m = gid / K_groups, g = gid % K_groups, k_start = g * 32;
    auto bf2f = [](unsigned short b) -> float { unsigned int f=(unsigned int)b<<16; return *reinterpret_cast<float*>(&f); };
    float amax = 0.0f;
    const unsigned short* row = A + m * K + k_start;
    for (int i = 0; i < 32; i++) { float v=bf2f(row[i]); float a=v<0?-v:v; if(a>amax)amax=a; }
    union { float f; unsigned int i; } u; u.f = amax;
    unsigned int amax_rounded = (u.i + 0x200000u) & 0xFF800000u; u.i = amax_rounded;
    int exp_bits = (amax_rounded >> 23) & 0xFF;
    int scale_unbiased = exp_bits - 127 - 2;
    int scale_biased = scale_unbiased + 127;
    if (scale_biased < 0) scale_biased = 0;
    unsigned char e8m0 = (unsigned char)scale_biased;
    A_scale[m * K_groups + g] = e8m0;
    int qs_exp = 127 - scale_unbiased;
    if (qs_exp < 1) qs_exp = 0; if (qs_exp > 254) qs_exp = 254;
    union { float f2; unsigned int i2; } qs; qs.i2 = (unsigned int)qs_exp << 23;
    float quant_scale = qs.f2;
    if (amax == 0.0f) quant_scale = 0.0f;
    const unsigned int denorm_exp = 149;
    const unsigned int denorm_mask_int = denorm_exp << 23;
    union { unsigned int di; float df; } dmu; dmu.di = denorm_mask_int;
    const float denorm_mask_float = dmu.df;
    const int val_to_add = ((1 - 127) << 23) + (1 << 21) - 1;
    unsigned char packed[16];
    for (int i = 0; i < 16; i++) {
        auto to_fp4 = [&](unsigned short bf16) -> unsigned char {
            float qx_f = bf2f(bf16) * quant_scale;
            unsigned int qx; { union { float f; unsigned int i; } t; t.f = qx_f; qx = t.i; }
            unsigned int s = qx & 0x80000000u; qx = qx ^ s;
            float qx_pos; { union { unsigned int i; float f; } t; t.i = qx; qx_pos = t.f; }
            unsigned char result;
            if (qx_pos >= 6.0f) { result = 0x7; }
            else if (qx_pos < 1.0f) {
                float dx = qx_pos + denorm_mask_float;
                unsigned int dxi; { union { float f; unsigned int i; } t; t.f = dx; dxi = t.i; }
                dxi -= denorm_mask_int;
                result = (unsigned char)(dxi & 0xFF);
            } else {
                unsigned int nx = qx;
                unsigned int mant_odd = (nx >> 22) & 1;
                nx = (unsigned int)((int)nx + val_to_add);
                nx += mant_odd;
                nx = nx >> 22;
                result = (unsigned char)(nx & 0xFF);
            }
            return (unsigned char)((result & 0x7) | ((s >> 28) & 0x8));
        };
        unsigned char lo = to_fp4(row[2*i]);
        unsigned char hi = to_fp4(row[2*i+1]);
        packed[i] = lo | (hi << 4);
    }
    unsigned char* out_row = A_q + m * K_half + g * 16;
    for (int i = 0; i < 16; i++) out_row[i] = packed[i];
}
)";
void init() {
    hiprtcProgram prog; CHRTC(hiprtcCreateProgram(&prog,src,"q.hip",0,nullptr,nullptr));
    const char* opts[]={"--offload-arch=gfx950","-O3"};
    CHRTC(hiprtcCompileProgram(prog,2,opts));
    size_t sz; CHRTC(hiprtcGetCodeSize(prog,&sz));
    char* code=new char[sz]; CHRTC(hiprtcGetCode(prog,code));
    CHIP(hipModuleLoadData(&g_mod,code)); CHIP(hipModuleGetFunction(&g_fn,g_mod,"mxfp4_quant"));
    delete[] code; hiprtcDestroyProgram(&prog);
}
void run_quant(torch::Tensor A, torch::Tensor Aq, torch::Tensor As) {
    int M=A.size(0),K=A.size(1),Kh=K/2,Kg=K/32;
    struct{void*A;void*Aq;void*As;int M;int K;int Kh;int Kg;} args;
    args.A=A.data_ptr();args.Aq=Aq.data_ptr();args.As=As.data_ptr();
    args.M=M;args.K=K;args.Kh=Kh;args.Kg=Kg;
    size_t sz=sizeof(args);
    void* config[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,&args,HIP_LAUNCH_PARAM_BUFFER_SIZE,&sz,HIP_LAUNCH_PARAM_END};
    int tot=M*Kg,thr=256,blk=(tot+thr-1)/thr;
    CHIP(hipModuleLaunchKernel(g_fn,blk,1,1,thr,1,1,0,0,nullptr,config));
    CHIP(hipDeviceSynchronize());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("init",&init);m.def("run_quant",&run_quant);}
"""
mod = load_inline(name="dbg_quant",cpp_sources=[hiprtc_quant_code],
                  extra_include_paths=["/opt/rocm/include"],
                  extra_ldflags=["-L/opt/rocm/lib","-lhiprtc","-lamdhip64"],
                  verbose=False,is_python_module=True)
mod.init()

# Compare quant outputs for M=256 K=1536
A = torch.randn(256, 1536, dtype=torch.bfloat16, device="cuda")

# Reference quant
A_q_ref, A_scale_ref = dynamic_mxfp4_quant(A)
A_scale_ref_c = A_scale_ref.contiguous()

# hiprtc quant
A_q_hip = torch.empty(256, 768, dtype=torch.uint8, device="cuda")
A_scale_hip = torch.empty(256, 48, dtype=torch.uint8, device="cuda")
mod.run_quant(A, A_q_hip, A_scale_hip)

# Compare scales
scale_diff = (A_scale_ref_c != A_scale_hip).sum().item()
print(f"Scale diff: {scale_diff}/{A_scale_ref_c.numel()}")

# Compare A_q
aq_diff = (A_q_ref != A_q_hip).sum().item()
print(f"A_q diff: {aq_diff}/{A_q_ref.numel()}")

if aq_diff > 0:
    # Show first few diffs
    mask = A_q_ref != A_q_hip
    idxs = mask.nonzero()[:10]
    for idx in idxs:
        i, j = idx[0].item(), idx[1].item()
        print(f"  [{i},{j}]: ref={A_q_ref[i,j].item()} hip={A_q_hip[i,j].item()}")

# Standard preshuffle kernel
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

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
