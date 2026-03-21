"""
MXFP4-MM: Triton GEMM with preshuffle API.
Uses pre-shuffled B_shuffle and B_scale_sh directly.
FAILED: Triton compilation error - NameError('b is not defined') when EVEN_K is False.
The preshuffle kernel has a bug in the non-EVEN_K code path.
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from aiter.utility.fp4_utils import e8m0_shuffle


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape

    # Quantize A
    A_q, A_scale = dynamic_mxfp4_quant(A)

    # For M >= 32, preshuffle expects shuffled A_scale
    # For M < 32, preshuffle expects unshuffled A_scale
    if m >= 32:
        A_scale_sh = e8m0_shuffle(A_scale)
    else:
        A_scale_sh = A_scale

    # Use preshuffle API with pre-shuffled B_shuffle and B_scale_sh
    return gemm_afp4wfp4_preshuffle(
        A_q.view(torch.uint8),
        B_shuffle.view(torch.uint8),
        A_scale_sh.view(torch.uint8),
        B_scale_sh.view(torch.uint8),
        dtype=torch.bfloat16,
    )
