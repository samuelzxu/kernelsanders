"""
MXFP4-MM: Pure CK (aiter.gemm_a4w4) for all shapes.
Baseline comparison to measure CK vs Triton on this runner.
"""
from task import input_t, output_t
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)

    return aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2), B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )
