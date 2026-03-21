"""
MXFP4-MM: #177 - Exact reference implementation.

Matches reference.py exactly:
  dynamic_mxfp4_quant(A) + e8m0_shuffle(A_scale) + gemm_a4w4(bpreshuffle=True)

The reference NOTE says to use dynamic_mxfp4_quant (patched #975) rather than
aiter.get_triton_quant which may dispatch to the unpatched fp4_utils.py kernel.
"""
import torch
from task import input_t, output_t
from aiter import dtypes
import aiter
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    # Exact reference path: dynamic_mxfp4_quant + e8m0_shuffle
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)

    return aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2),
        B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
