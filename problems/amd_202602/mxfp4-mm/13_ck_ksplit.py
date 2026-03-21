"""
MXFP4-MM: Hybrid CK/Triton with log2_k_split on CK for large K.
Small K uses Triton (faster), large K uses CK ASM with split-K.
"""
from task import input_t, output_t
import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape

    A_q, A_scale = dynamic_mxfp4_quant(A)

    if k <= 512:
        # Triton path for small K
        B_q_raw, B_scale = dynamic_mxfp4_quant(B)
        return gemm_afp4wfp4(A_q, B_q_raw, A_scale, B_scale, dtype=torch.bfloat16)
    else:
        # CK path with split-K for large K
        A_scale_sh = e8m0_shuffle(A_scale)
        # log2_k_split: 1 = 2-way split, 2 = 4-way split
        log2_ks = 2 if k >= 4096 else 1
        return aiter.gemm_a4w4(
            A_q.view(dtypes.fp4x2), B_shuffle,
            A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
            dtype=dtypes.bf16, bpreshuffle=True,
            log2_k_split=log2_ks,
        )
