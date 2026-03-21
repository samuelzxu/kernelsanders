"""
MXFP4-MM: #176 - Correct baseline using aiter.get_triton_quant + gemm_a4w4.

The problem description explicitly states:
  (1) Quant A to MXFP4: aiter.get_triton_quant(QuantType.per_1x32)
  (2) GEMM: aiter.gemm_a4w4

We've been using dynamic_mxfp4_quant + gemm_afp4wfp4 (Triton path).
get_triton_quant may return A_q/A_scale already in shuffled format,
eliminating the e8m0_shuffle overhead that killed #164.
"""
import torch
from task import input_t, output_t
from aiter import dtypes, QuantType
import aiter

quant_func = aiter.get_triton_quant(QuantType.per_1x32)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    A_q, A_scale = quant_func(A)
    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale, B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
