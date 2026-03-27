#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#504: Combined quant + CK ASM GEMM in single C++ call.
Zero Python round-trips between quant and GEMM.
"""
import torch, os, json
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
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

# Combined quant + CK ASM GEMM in single C++ function
_hip_src = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// ===== Fused quant kernel (same as before) =====
extern "C" __global__ void fused_quant_shuffle(
    const unsigned short* __restrict__ A,
    unsigned char* __restrict__ A_q,
    unsigned char* __restrict__ A_scale_sh,
    int M, int K, int K_half, int K_groups, int M_pad
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = M_pad * K_groups;
    if (gid >= total_groups) return;
    int m = gid / K_groups;
    int g = gid % K_groups;

    float vals[32]; float amax = 0.0f;
    if (m < M) {
        const uint4* row4 = (const uint4*)(A + m * K + g * 32);
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            uint4 chunk = row4[v];
            unsigned int w[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float lo = __uint_as_float((w[j] & 0xFFFFu) << 16);
                float hi = __uint_as_float(w[j] & 0xFFFF0000u);
                vals[v*8+j*2] = lo; vals[v*8+j*2+1] = hi;
                amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
            }
        }
    } else { for(int i=0;i<32;i++) vals[i]=0.0f; }

    unsigned int ai = __float_as_uint(amax);
    unsigned int ar = (ai + 0x200000u) & 0xFF800000u;
    int eb = (ar >> 23) & 0xFF;
    int su = eb - 129;
    int sb = su + 127;
    if (sb < 0) sb = 0;

    // E8M0 shuffle write
    int m0=m/32; int m1=(m&31)/16; int m2=m&15;
    int g0=g/8; int g1=(g&7)/4; int g2=g&3;
    int kg8 = K_groups / 8;
    A_scale_sh[m0*(kg8*256)+g0*256+g2*64+m2*4+g1*2+m1] = (unsigned char)sb;

    int qe = 127-su; if(qe<1)qe=0; if(qe>254)qe=254;
    float qs = __uint_as_float((unsigned int)qe << 23);
    if (amax == 0.0f) qs = 0.0f;

    const unsigned int dmi = 149u << 23;
    float dmf = __uint_as_float(dmi);
    const int vta = ((int)(1-127) << 23) + (1 << 21) - 1;

    auto fp4 = [&](float v) -> unsigned char {
        float qf = v * qs;
        unsigned int qx = __float_as_uint(qf);
        unsigned int s = qx & 0x80000000u;
        qx ^= s;
        float qp = __uint_as_float(qx);
        unsigned char r;
        if (qp >= 6.0f) r = 0x7;
        else if (qp < 1.0f) r = (unsigned char)((__float_as_uint(qp + dmf) - dmi) & 0xFF);
        else { unsigned int mo = (qx >> 22) & 1; r = (unsigned char)((((unsigned int)((int)qx + vta) + mo) >> 22) & 0xFF); }
        return (r & 0x7) | ((unsigned char)(s >> 28) & 0x8);
    };

    unsigned char packed[16];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        packed[i] = fp4(vals[2*i]) | (fp4(vals[2*i+1]) << 4);
    uint4* out4 = (uint4*)(A_q + m * K_half + g * 16);
    *out4 = *((uint4*)packed);
}

// ===== CK ASM GEMM launcher =====
struct __attribute__((packed)) KArgs {
    void* ptr_D;     char _p0[8];
    void* ptr_C;     char _p1[8];
    void* ptr_A;     char _p2[8];
    void* ptr_B;     char _p3[8];
    float alpha;     char _p4[12];
    float beta;      char _p5[12];
    unsigned int stride_D0; char _p6[12];
    unsigned int stride_D1; char _p7[12];
    unsigned int stride_C0; char _p8[12];
    unsigned int stride_C1; char _p9[12];
    unsigned int stride_A0; char _p10[12];
    unsigned int stride_A1; char _p11[12];
    unsigned int stride_B0; char _p12[12];
    unsigned int stride_B1; char _p13[12];
    unsigned int M;  char _p14[12];
    unsigned int N;  char _p15[12];
    unsigned int K;  char _p16[12];
    void* ptr_ScaleA; char _p17[8];
    void* ptr_ScaleB; char _p18[8];
    unsigned int stride_ScaleA0; char _p19[12];
    unsigned int stride_ScaleA1; char _p20[12];
    unsigned int stride_ScaleB0; char _p21[12];
    unsigned int stride_ScaleB1; char _p22[12];
    int log2_k_split;
};

static hipModule_t g_ck_mod = nullptr;
static hipFunction_t g_ck_fn = nullptr;
static torch::Tensor g_aq, g_ash, g_out;
static int g_M_pad = 0, g_K = 0, g_N = 0;

bool init_ck_kernel(const std::string& co_path, const std::string& fn_name) {
    if (g_ck_fn) return true;
    if (hipModuleLoad(&g_ck_mod, co_path.c_str()) != hipSuccess) return false;
    if (hipModuleGetFunction(&g_ck_fn, g_ck_mod, fn_name.c_str()) != hipSuccess) {
        g_ck_mod = nullptr; return false;
    }
    return true;
}

// Combined: quant A → CK ASM GEMM → return bf16 output
// Single Python→C++ call, zero round-trips
torch::Tensor quant_and_gemm(
    torch::Tensor A,          // [M, K] bf16
    torch::Tensor B_shuffle,  // preshuffle FP4
    torch::Tensor B_scale_sh, // shuffled E8M0
    int N_val
) {
    if (!g_ck_fn) return torch::Tensor();

    int M = A.size(0), K_elem = A.size(1);
    int M_pad = ((M + 31) / 32) * 32;
    int K_half = K_elem / 2;
    int K_groups = K_elem / 32;

    // Allocate quant buffers (reuse if same shape)
    if (M_pad != g_M_pad || K_elem != g_K) {
        g_aq = torch::empty({M_pad, K_half}, torch::dtype(torch::kUInt8).device(A.device()));
        g_ash = torch::empty({M_pad, K_groups}, torch::dtype(torch::kUInt8).device(A.device()));
        g_M_pad = M_pad; g_K = K_elem;
    }
    if (N_val != g_N || M_pad != g_M_pad) {
        g_out = torch::empty({M_pad, N_val}, torch::dtype(torch::kBFloat16).device(A.device()));
        g_N = N_val;
    }

    // 1) Launch quant kernel
    int total_groups = M_pad * K_groups;
    int threads = 256;
    int qblocks = (total_groups + threads - 1) / threads;
    fused_quant_shuffle<<<qblocks, threads, 0, 0>>>(
        (const unsigned short*)A.data_ptr(),
        g_aq.data_ptr<unsigned char>(),
        g_ash.data_ptr<unsigned char>(),
        M, K_elem, K_half, K_groups, M_pad
    );

    // 2) Launch CK ASM GEMM immediately (same HIP queue, no sync needed)
    KArgs ka;
    memset(&ka, 0, sizeof(ka));
    ka.ptr_D = g_out.data_ptr();
    ka.ptr_C = g_out.data_ptr();
    ka.ptr_A = g_aq.data_ptr<unsigned char>();
    ka.ptr_B = B_shuffle.data_ptr();
    ka.alpha = 1.0f; ka.beta = 0.0f;
    ka.stride_D0 = N_val; ka.stride_D1 = 1;
    ka.stride_C0 = N_val; ka.stride_C1 = 1;
    ka.stride_A0 = K_elem; ka.stride_A1 = 1;  // FP4x2: stride = K/2 * 2 = K
    ka.stride_B0 = B_shuffle.stride(0) * 2; ka.stride_B1 = 1;
    ka.M = M_pad; ka.N = N_val; ka.K = K_elem;
    ka.ptr_ScaleA = g_ash.data_ptr<unsigned char>();
    ka.ptr_ScaleB = B_scale_sh.data_ptr();
    ka.stride_ScaleA0 = K_groups; ka.stride_ScaleA1 = 1;
    ka.stride_ScaleB0 = B_scale_sh.stride(0); ka.stride_ScaleB1 = 1;
    ka.log2_k_split = 0;

    size_t asz = sizeof(ka);
    void* cfg[] = {(void*)0x01, &ka, (void*)0x02, &asz, (void*)0x03};
    // 32x128 tile: grid = (ceil(N/128), ceil(M/32), 1)
    int gx = (N_val + 127) / 128;
    int gy = (M_pad + 31) / 32;
    hipModuleLaunchKernel(g_ck_fn, gx, gy, 1, 256, 1, 1, 0, 0, nullptr, cfg);

    return g_out.slice(0, 0, M);
}
"""

_cpp_src = r"""
#include <torch/extension.h>
bool init_ck_kernel(const std::string& co_path, const std::string& fn_name);
torch::Tensor quant_and_gemm(torch::Tensor A, torch::Tensor B_shuffle, torch::Tensor B_scale_sh, int N_val);
"""

_combined_mod = None
try:
    _combined_mod = load_inline(
        name='combined_504',
        cpp_sources=_cpp_src,
        cuda_sources=_hip_src,
        functions=['init_ck_kernel', 'quant_and_gemm'],
        verbose=False,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    _co = "/home/runner/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.co"
    _fn = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
    if _combined_mod.init_ck_kernel(_co, _fn):
        # Warmup
        _wA = torch.randn(256, 1536, dtype=torch.bfloat16, device="cuda")
        _wBq = torch.zeros(3072, 768, dtype=torch.uint8, device="cuda")
        _wBs = torch.zeros(3072, 48, dtype=torch.uint8, device="cuda")
        _combined_mod.quant_and_gemm(_wA, _wBq, _wBs, 3072)
        torch.cuda.synchronize()
        del _wA, _wBq, _wBs
        print("Combined quant+CK ASM OK")
    else:
        _combined_mod = None
        print("CK kernel init failed")
except Exception as e:
    _combined_mod = None
    print(f"Combined load_inline FAILED: {e}")

# Warmup gemm_a4w4 as fallback
for _m,_n,_k in [(256,3072,1536)]:
    try:
        _A=torch.randn(_m,_k,dtype=torch.bfloat16,device="cuda")
        _aq,_as=dynamic_mxfp4_quant(_A)
        _aq=_aq.view(dtypes.fp4x2)
        _as=e8m0_shuffle(_as).view(dtypes.fp8_e8m0)
        from aiter.ops.shuffle import shuffle_weight
        _bq=torch.zeros(_n,_k//2,dtype=torch.uint8,device="cuda").view(dtypes.fp4x2)
        _bsh=shuffle_weight(_bq,layout=(16,16))
        _bsc=torch.zeros(_n,_k//32,dtype=torch.uint8,device="cuda").view(dtypes.fp8_e8m0)
        aiter.gemm_a4w4(_aq,_bsh,_as,_bsc,dtype=dtypes.bf16,bpreshuffle=True)
    except:pass
torch.cuda.empty_cache()

_ps_ck=None;_ps_cw=None;_ps_cs=None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]

    # Combined quant+GEMM in single C++ call for K=1536 M=256
    if _combined_mod is not None and m == 256 and k == 1536:
        return _combined_mod.quant_and_gemm(A, B_shuffle, B_scale_sh, n)

    # Preshuffle fallback
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
