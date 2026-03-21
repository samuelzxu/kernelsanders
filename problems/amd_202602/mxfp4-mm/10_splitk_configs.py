"""
MXFP4-MM: Triton GEMM with per-shape custom configs.
Uses gemm_afp4wfp4 with split-K for large K cases.
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


# Per-shape configs: (M_threshold, K_threshold) -> config
# Config: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT
def _get_custom_config(m, n, k):
    # K here is K//2 (fp4 packed), so actual_K = k * 2
    # For large K, use split-K to parallelize
    if k >= 2048:  # actual K >= 4096 (e.g., K=7168)
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 4,
        }
    elif k >= 512:  # actual K >= 1024 (e.g., K=2048, 1536)
        return {
            "BLOCK_SIZE_M": 32 if m >= 32 else 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 2,
        }
    else:  # small K (e.g., K=512)
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 1,
        }


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A
    A_q, A_scale = dynamic_mxfp4_quant(A)

    # Re-quantize B to get unshuffled B_q and B_scale for Triton
    B_q_raw, B_scale = dynamic_mxfp4_quant(B)

    # Get custom config for this shape
    config = _get_custom_config(m, n, k)

    return gemm_afp4wfp4(
        A_q, B_q_raw, A_scale, B_scale,
        dtype=torch.bfloat16,
        config=config,
    )
