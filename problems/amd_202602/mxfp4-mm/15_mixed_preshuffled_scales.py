"""
MXFP4-MM: Mixed approach.
- M >= 32: gemm_afp4wfp4_preshuffled_scales (different scale layout)
- M < 32: gemm_afp4wfp4 (regular Triton)
Both avoid double quantization via e8m0_unshuffle.
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffled_scales,
)
from aiter.utility.fp4_utils import e8m0_shuffle


def e8m0_unshuffle(scale, orig_m, orig_n):
    """Reverse e8m0_shuffle: permute back and unpad."""
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 4, 2, 1).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    A_q, A_scale = dynamic_mxfp4_quant(A)

    B_q_uint8 = B_q.view(torch.uint8)

    if m >= 32:
        # Use preshuffled_scales variant with shuffled A/B scales
        A_scale_sh = e8m0_shuffle(A_scale)
        # preshuffled_scales expects: x(M,K), w(N,K), x_scales(M//32,K), w_scales(N//32,K)
        return gemm_afp4wfp4_preshuffled_scales(
            A_q, B_q_uint8, A_scale_sh.view(torch.uint8), B_scale_sh.view(torch.uint8),
            dtype=torch.bfloat16,
        )
    else:
        # Regular path with unshuffled scales
        B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        return gemm_afp4wfp4(
            A_q, B_q_uint8, A_scale, B_scale,
            dtype=torch.bfloat16,
        )
