"""
MXFP4-MM: Triton GEMM with corrected e8m0_unshuffle.
Avoids double B quantization by unshuffling pre-computed B_scale_sh.
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def e8m0_unshuffle(scale, orig_m, orig_n):
    """Reverse e8m0_shuffle: inverse permute and unpad."""
    sm, sn = scale.shape
    # Forward shuffle: view(sm//32, 2, 16, sn//8, 2, 4).permute(0, 3, 5, 2, 4, 1)
    # Result shape: (sm//32, sn//8, 4, 16, 2, 2)
    # Inverse permute: (0, 5, 3, 1, 4, 2)
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    n, k = B.shape

    # Quantize A (always needed)
    A_q, A_scale = dynamic_mxfp4_quant(A)

    # Reuse pre-computed B_q (view as uint8 for Triton)
    B_q_uint8 = B_q.view(torch.uint8)

    # Unshuffle B_scale_sh to get original B_scale
    B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)

    return gemm_afp4wfp4(
        A_q, B_q_uint8, A_scale, B_scale,
        dtype=torch.bfloat16,
    )
