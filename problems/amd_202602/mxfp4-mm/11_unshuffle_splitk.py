"""
MXFP4-MM: e8m0_unshuffle + split-K configs.
Avoids double quantization by unshuffling B_scale, and uses split-K for large K.
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def e8m0_unshuffle(scale, orig_m, orig_n):
    """Reverse e8m0_shuffle: permute back and unpad."""
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 4, 2, 1).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def _get_config(m, n, k):
    """Custom config based on shape. k is actual K (not packed)."""
    if k >= 4096:  # K=7168
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 4,
        }
    elif k >= 1024:  # K=2048, 1536
        return {
            "BLOCK_SIZE_M": 32 if m >= 32 else 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 2,
        }
    else:  # K=512
        return None  # Use default config


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A
    A_q, A_scale = dynamic_mxfp4_quant(A)

    # Reuse pre-computed B_q and unshuffle B_scale
    B_q_uint8 = B_q.view(torch.uint8)
    B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)

    config = _get_config(m, n, k)

    return gemm_afp4wfp4(
        A_q, B_q_uint8, A_scale, B_scale,
        dtype=torch.bfloat16,
        config=config,
    )
