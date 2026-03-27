#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#294: Deep probe gemm_a4w4_blockscale and _dynamic_mxfp4_quant_kernel_asm_layout.
"""
import torch, sys, inspect, time
from task import input_t, output_t

# Probe 1: gemm_a4w4_blockscale source/signature
print("=== gemm_a4w4_blockscale ===")
try:
    from aiter import gemm_a4w4_blockscale
    src = inspect.getsource(gemm_a4w4_blockscale)
    print(src[:3000])
except Exception as e:
    print(f"Error: {e}")

# Probe 2: _dynamic_mxfp4_quant_kernel_asm_layout
print("\n=== _dynamic_mxfp4_quant_kernel_asm_layout ===")
try:
    from aiter.fp4_utils import _dynamic_mxfp4_quant_kernel_asm_layout
    fn = _dynamic_mxfp4_quant_kernel_asm_layout
    if hasattr(fn, 'fn'):
        fn = fn.fn
    if hasattr(fn, 'fn'):
        fn = fn.fn
    src = inspect.getsource(fn)
    print(src[:3000])
except Exception as e:
    print(f"Error: {e}")

# Probe 3: dynamic_per_group_scaled_quant_fp4
print("\n=== dynamic_per_group_scaled_quant_fp4 ===")
try:
    from aiter import dynamic_per_group_scaled_quant_fp4
    src = inspect.getsource(dynamic_per_group_scaled_quant_fp4)
    print(src[:2000])
except Exception as e:
    print(f"Error: {e}")

# Probe 4: gemm_a4w4_blockscale_tune
print("\n=== gemm_a4w4_blockscale_tune ===")
try:
    from aiter import gemm_a4w4_blockscale_tune
    src = inspect.getsource(gemm_a4w4_blockscale_tune)
    print(src[:2000])
except Exception as e:
    print(f"Error: {e}")

# Probe 5: Try calling gemm_a4w4_blockscale with test data
print("\n=== gemm_a4w4_blockscale test ===")
try:
    from aiter import gemm_a4w4_blockscale
    from aiter.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle
    M, N, K = 4, 2880, 512
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    A_q, A_scale = dynamic_mxfp4_quant(A)
    B_q = torch.zeros((N, K//2), dtype=torch.uint8, device="cuda")
    B_scale = torch.zeros((N, K//32), dtype=torch.uint8, device="cuda")
    # Try different call patterns
    try:
        out = gemm_a4w4_blockscale(A_q, B_q, A_scale, B_scale)
        print(f"Call 1 success: {out.shape}")
    except Exception as e:
        print(f"Call 1 error: {e}")
    try:
        A_scale_sh = e8m0_shuffle(A_scale).contiguous()
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        out = gemm_a4w4_blockscale(A_q, B_q, A_scale_sh, B_scale, out)
        print(f"Call 2 success: {out.shape}")
    except Exception as e:
        print(f"Call 2 error: {e}")
except Exception as e:
    print(f"Setup error: {e}")

# Probe 6: Check if there's a way to call preshuffle ASM directly
print("\n=== gemm_a4w4 with preshuffle ===")
try:
    from aiter import gemm_a4w4
    src = inspect.getsource(gemm_a4w4)
    print(src[:3000])
except Exception as e:
    print(f"Error: {e}")

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
_ck = None; _cw = None; _cs = None
@torch.inference_mode()
def custom_kernel(data):
    global _ck, _cw, _cs
    A = data[0]; B_shuffle = data[3]; B_scale_sh = data[4]
    m, k = A.shape; n = data[1].shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _ck:
        _ck = dp
        _cw = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _cs = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _cw, _cs, prequant=True, dtype=torch.bfloat16)
