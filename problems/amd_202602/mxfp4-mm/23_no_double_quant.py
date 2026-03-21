"""
MXFP4-MM: Hybrid CK/Triton - no double quantization for any shape.
- K <= 512: CK (aiter.gemm_a4w4) with pre-shuffled B, shuffled scales
- K > 512: Triton with e8m0_unshuffle (avoids double B quant)
Eliminates dynamic_mxfp4_quant(B) entirely.
"""
from task import input_t, output_t
import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
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
        # CK path: uses shuffled data directly, no B re-quantization
        A_scale_sh = e8m0_shuffle(A_scale)
        return aiter.gemm_a4w4(
            A_q.view(dtypes.fp4x2), B_shuffle,
            A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
            dtype=dtypes.bf16, bpreshuffle=True,
        )
    else:
        # Triton path: unshuffle B_scale to avoid B re-quantization
        B_q_uint8 = B_q.view(torch.uint8)
        B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
