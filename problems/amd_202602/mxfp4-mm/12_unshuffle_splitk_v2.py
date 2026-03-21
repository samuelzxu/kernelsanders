"""
MXFP4-MM: e8m0_unshuffle + modified default configs with split-K.
"""
from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config


def e8m0_unshuffle(scale, orig_m, orig_n):
    """Reverse e8m0_shuffle: permute back and unpad."""
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 4, 2, 1).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A
    A_q, A_scale = dynamic_mxfp4_quant(A)

    # Reuse pre-computed B_q and unshuffle B_scale
    B_q_uint8 = B_q.view(torch.uint8)
    B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)

    # Get default config and enable split-K for large K
    config, _ = _get_config(m, n, k)
    if k >= 4096:
        config["NUM_KSPLIT"] = 4
    elif k >= 1024:
        config["NUM_KSPLIT"] = 2

    return gemm_afp4wfp4(
        A_q, B_q_uint8, A_scale, B_scale,
        dtype=torch.bfloat16,
        config=config,
    )
