"""
MXFP4-MM: Hybrid CK/Triton GEMM - best result so far (~20.4µs geomean).
K<=512 uses Triton (no shuffle overhead), K>512 uses CK.
Submission b81806b results:
  K=512,  M=4,   N=2880: 15.8µs
  K=7168, M=16,  N=2112: 34.4µs
  K=512,  M=32,  N=4096: 14.9µs
  K=512,  M=32,  N=2880: 15.1µs
  K=2048, M=64,  N=7168: 25.0µs
  K=1536, M=256, N=3072: 23.5µs
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
    k = A.shape[1]

    A_q, A_scale = dynamic_mxfp4_quant(A)

    if k <= 512:
        # Triton path: re-quantize B to get unshuffled tensors
        B_q_raw, B_scale = dynamic_mxfp4_quant(B)
        return gemm_afp4wfp4(A_q, B_q_raw, A_scale, B_scale, dtype=torch.bfloat16)
    else:
        # CK path: use pre-shuffled B and shuffled scales
        A_scale_sh = e8m0_shuffle(A_scale)
        return aiter.gemm_a4w4(
            A_q.view(dtypes.fp4x2), B_shuffle,
            A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
            dtype=dtypes.bf16, bpreshuffle=True,
        )
