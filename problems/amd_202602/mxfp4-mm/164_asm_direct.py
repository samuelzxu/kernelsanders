"""
MXFP4-MM: #164 - Route all shapes through aiter ASM GEMM kernels.

The reference kernel uses aiter.gemm_a4w4(bpreshuffle=True) which dispatches
to pre-compiled hand-optimized ASM kernels (.co files). These have LDS tiling,
double buffering, and ping-pong scheduling — fundamentally faster than Triton.

From the tuned CSV (a4w4_blockscale_tuned_gemm.csv):
  M=1,  N=2112, K=7168: 12.36µs (vs Triton ~15.5µs)
  M=8,  N=7168, K=2048:  5.84µs (vs Triton ~13.7µs)
  M=64, N=7168, K=2048:  6.81µs (vs Triton ~13.7µs)

Problem: ranked shapes (M=4,N=2880,K=512), (M=16,N=2112,K=7168), etc. are
NOT in the CSV, so gemm_a4w4 falls back to a slow default. Fix: monkey-patch
the dispatch table to inject ASM kernel selections for our exact shapes.

Kernel choices from closest CSV entries:
  BpreShuffle_32x128 (id=21): Best for M<=64 (most versatile, low overhead)
  BpreShuffle_64x128 (id=29): Good for M=64-128
  BpreShuffle_192x128 (id=13): Good for M=128-256
"""
import os
import torch
from task import input_t, output_t
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.gemm_op_a4w4 import get_GEMM_config
import aiter

# Force-load the CSV dispatch table, then inject our entries
_dummy = get_GEMM_config(1, 1, 64)  # triggers CSV load

# kernel names from the .co files
_K32x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
_K64x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"
_K192x128 = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128E"
_K256x128 = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128E"

# Inject configs for our ranked shapes
# Format: (cu_num, padded_M, N, K) → {kernelId, splitK, kernelName, ...}
_INJECT = {
    # k=512, m=4, n=2880
    (256, 4, 2880, 512): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    (256, 8, 2880, 512): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    # k=7168, m=16, n=2112
    (256, 16, 2112, 7168): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    # k=512, m=32, n=4096
    (256, 32, 4096, 512): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    # k=512, m=32, n=2880
    (256, 32, 2880, 512): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    # k=2048, m=64, n=7168
    (256, 64, 7168, 2048): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    # k=1536, m=256, n=3072
    (256, 256, 3072, 1536): {"kernelId": 21, "splitK": 0, "kernelName": _K256x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    # Also add for test shapes
    (256, 8, 2112, 7168): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    (256, 16, 3072, 1536): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    (256, 64, 3072, 1536): {"kernelId": 21, "splitK": 0, "kernelName": _K32x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
    (256, 256, 2880, 512): {"kernelId": 21, "splitK": 0, "kernelName": _K256x128, "us": 0, "tflops": 0, "bw": 0, "errRatio": 0},
}

if hasattr(get_GEMM_config, "gemm_dict"):
    get_GEMM_config.gemm_dict.update(_INJECT)
    print(f"[INJECT] Added {len(_INJECT)} ASM kernel configs", file=__import__('sys').stderr)


_out_cache = {}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Quantize A to MXFP4 and shuffle scale
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)

    # Call ASM GEMM via aiter.gemm_a4w4 with preshuffled data
    result = aiter.gemm_a4w4(
        A_q.view(dtypes.fp4x2),
        B_shuffle,
        A_scale_sh.view(dtypes.fp8_e8m0),
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
    return result
