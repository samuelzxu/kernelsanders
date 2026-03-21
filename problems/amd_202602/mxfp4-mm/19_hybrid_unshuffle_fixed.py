"""
MXFP4-MM: Hybrid Triton with corrected e8m0_unshuffle.
- K <= 512: re-quantize B (cheaper than unshuffle for small K)
- K > 512: unshuffle B_scale (avoids expensive B re-quantization)
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def e8m0_unshuffle(scale, orig_m, orig_n):
    """Reverse e8m0_shuffle: inverse permute and unpad."""
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A (always needed)
    A_q, A_scale = dynamic_mxfp4_quant(A)

    if k <= 512:
        # Small K: re-quantize B is cheaper than unshuffle
        B_q_raw, B_scale = dynamic_mxfp4_quant(B)
        return gemm_afp4wfp4(A_q, B_q_raw, A_scale, B_scale, dtype=torch.bfloat16)
    else:
        # Large K: unshuffle B_scale to avoid expensive B re-quantization
        B_q_uint8 = B_q.view(torch.uint8)
        B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
