#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#462: Fused quant+e8m0_shuffle via load_inline + gemm_a4w4 for M>=32.
Preshuffle for M<=16. Within-call quant (no cross-invocation caching).
"""
import torch, os, json
from task import input_t, output_t
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from torch.utils.cpp_extension import load_inline

# === Preshuffle setup (for M<=16 shapes) ===
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={
    "N=2880-K=512":{
        "M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=4096-K=512":{
        "M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=2112-K=7168":{
        "M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=7168-K=2048":{
        "M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}
    },
    "N=3072-K=1536":{
        "M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},
        "M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},
        "any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}
    }
}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warmup preshuffle
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
torch.cuda.empty_cache()

# === Fused quant+e8m0_shuffle kernel via load_inline ===
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

_hip_src = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// Fused MXFP4 quant + e8m0_shuffle in one kernel
// Each thread handles one group of 32 bf16 elements
extern "C" __global__ void fused_quant_shuffle(
    const unsigned short* __restrict__ A,  // bf16 as uint16 [M, K]
    unsigned char* __restrict__ A_q,        // FP4 packed [M_pad, K/2]
    unsigned char* __restrict__ A_scale_sh, // E8M0 shuffled [M_pad, K/32]
    int M, int K, int K_half, int K_groups, int M_pad
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = M_pad * K_groups;
    if (gid >= total_groups) return;

    int m = gid / K_groups;
    int g = gid % K_groups;
    int k_start = g * 32;

    // bf16 to float
    auto bf2f = [](unsigned short b) -> float {
        unsigned int f = (unsigned int)b << 16;
        return *reinterpret_cast<float*>(&f);
    };

    // Read 32 bf16 elements (handle M padding)
    float vals[32];
    if (m < M) {
        const unsigned short* row = A + m * K + k_start;
        for (int i = 0; i < 32; i++) vals[i] = bf2f(row[i]);
    } else {
        for (int i = 0; i < 32; i++) vals[i] = 0.0f;
    }

    // Find max abs
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float abs_val = vals[i] < 0 ? -vals[i] : vals[i];
        if (abs_val > amax) amax = abs_val;
    }

    // E8M0 encoding
    unsigned int ai = __float_as_uint(amax);
    unsigned int amax_rounded = (ai + 0x200000u) & 0xFF800000u;
    int exp_bits = (amax_rounded >> 23) & 0xFF;
    int scale_unbiased = exp_bits - 127 - 2;
    int scale_biased = scale_unbiased + 127;
    if (scale_biased < 0) scale_biased = 0;
    unsigned char e8m0 = (unsigned char)scale_biased;

    // E8M0 shuffle permutation: write to shuffled position
    // view(M_pad//32, 2, 16, K_groups//8, 2, 4).permute(0,3,5,2,4,1)
    int m0 = m / 32;
    int m1 = (m & 31) / 16;
    int m2 = m & 15;
    int g0 = g / 8;
    int g1 = (g & 7) / 4;
    int g2 = g & 3;
    int kg8 = K_groups / 8;
    // Output shape: (M_pad//32, K_groups//8, 4, 16, 2, 2)
    // Strides: (kg8*256, 256, 64, 4, 2, 1)
    int out_idx = m0 * (kg8 * 256) + g0 * 256 + g2 * 64 + m2 * 4 + g1 * 2 + m1;
    A_scale_sh[out_idx] = e8m0;

    // Quant scale
    int qs_exp = 127 - scale_unbiased;
    if (qs_exp < 1) qs_exp = 0;
    if (qs_exp > 254) qs_exp = 254;
    float quant_scale = __uint_as_float((unsigned int)qs_exp << 23);
    if (amax == 0.0f) quant_scale = 0.0f;

    // FP4 quant constants
    const float max_normal = 6.0f;
    const float min_normal = 1.0f;
    const unsigned int denorm_mask_int = 149u << 23;  // (127-1+23-1+1)<<23
    float denorm_mask_float = __uint_as_float(denorm_mask_int);
    const int val_to_add = ((int)(1 - 127) << 23) + (1 << 21) - 1;

    // Pack 32 FP4 values into 16 bytes
    unsigned char* out_row = A_q + m * K_half + g * 16;
    for (int i = 0; i < 16; i++) {
        auto to_fp4 = [&](float qx_f) -> unsigned char {
            unsigned int qx = __float_as_uint(qx_f);
            unsigned int s = qx & 0x80000000u;
            qx = qx ^ s;
            float qx_pos = __uint_as_float(qx);
            unsigned char result;
            if (qx_pos >= max_normal) {
                result = 0x7;
            } else if (qx_pos < min_normal) {
                float denormal_x = qx_pos + denorm_mask_float;
                unsigned int dx = __float_as_uint(denormal_x) - denorm_mask_int;
                result = (unsigned char)(dx & 0xFF);
            } else {
                unsigned int mant_odd = (qx >> 22) & 1;
                qx = (unsigned int)((int)qx + val_to_add) + mant_odd;
                qx = qx >> 22;
                result = (unsigned char)(qx & 0xFF);
            }
            return (result & 0x7) | ((unsigned char)(s >> 28) & 0x8);
        };
        unsigned char lo = to_fp4(vals[2*i] * quant_scale);
        unsigned char hi = to_fp4(vals[2*i+1] * quant_scale);
        out_row[i] = lo | (hi << 4);
    }
}

// C++ wrapper
std::vector<torch::Tensor> fused_quant(torch::Tensor A) {
    int M = A.size(0), K = A.size(1);
    int M_pad = ((M + 31) / 32) * 32;
    int K_half = K / 2;
    int K_groups = K / 32;

    auto A_q = torch::empty({M_pad, K_half}, torch::dtype(torch::kUInt8).device(A.device()));
    auto A_scale_sh = torch::empty({M_pad, K_groups}, torch::dtype(torch::kUInt8).device(A.device()));

    int total_groups = M_pad * K_groups;
    int threads = 256;
    int blocks = (total_groups + threads - 1) / threads;

    fused_quant_shuffle<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const unsigned short*)A.data_ptr(),
        A_q.data_ptr<unsigned char>(),
        A_scale_sh.data_ptr<unsigned char>(),
        M, K, K_half, K_groups, M_pad
    );

    return {A_q, A_scale_sh};
}
"""

_cpp_src = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> fused_quant(torch::Tensor A);
"""

try:
    _quant_mod = load_inline(
        name='fused_quant_462',
        cpp_sources=_cpp_src,
        cuda_sources=_hip_src,
        functions=['fused_quant'],
        verbose=False,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    print("load_inline OK")
except Exception as e:
    _quant_mod = None
    print(f"load_inline FAILED: {e}")

# Warmup gemm_a4w4 for M>=32 shapes
for _m,_n,_k in [(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
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
    except Exception as e:
        print(f"Warmup a4w4 {_m}x{_n}x{_k}: {e}")
torch.cuda.empty_cache()

# Cache for preshuffle B reshape
_ps_ck=None;_ps_cw=None;_ps_cs=None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]

    # Use fused quant + gemm_a4w4 for large M shapes (if load_inline succeeded)
    if _quant_mod is not None and m >= 32:
        aq_raw, ash_raw = _quant_mod.fused_quant(A)
        aq = aq_raw.view(dtypes.fp4x2)
        ash = ash_raw.view(dtypes.fp8_e8m0)
        return aiter.gemm_a4w4(aq, B_shuffle, ash, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)

    # Preshuffle path for small M or fallback
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
