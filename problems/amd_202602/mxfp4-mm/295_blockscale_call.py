#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#295: Try calling gemm_a4w4_blockscale with various argument combinations.
Also try dynamic_per_group_scaled_quant_fp4.
"""
import torch, sys, time
from task import input_t, output_t

M, N, K = 256, 3072, 1536

# Create test data
A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
B_q = torch.zeros((N, K//2), dtype=torch.uint8, device="cuda")
B_scale = torch.zeros((N, K//32), dtype=torch.uint8, device="cuda")

# Quant A
from aiter.ops.triton.quant import dynamic_mxfp4_quant, e8m0_shuffle
A_q, A_scale = dynamic_mxfp4_quant(A)
A_scale_sh = e8m0_shuffle(A_scale).contiguous()

print(f"A_q: {A_q.shape} {A_q.dtype}")
print(f"A_scale: {A_scale.shape} {A_scale.dtype}")
print(f"A_scale_sh: {A_scale_sh.shape} {A_scale_sh.dtype}")
print(f"B_q: {B_q.shape} {B_q.dtype}")
print(f"B_scale: {B_scale.shape} {B_scale.dtype}")

# Try gemm_a4w4_blockscale
print("\n=== gemm_a4w4_blockscale attempts ===")
from aiter import gemm_a4w4_blockscale

# Attempt various signatures
out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

# Try 1: (A_q, B_q, A_scale, B_scale)
try:
    r = gemm_a4w4_blockscale(A_q, B_q, A_scale, B_scale)
    print(f"Try1 (A_q,B_q,A_s,B_s): {r.shape} {r.dtype}")
except Exception as e:
    print(f"Try1: {e}")

# Try 2: (A_q, B_q, A_scale, B_scale, out)
try:
    r = gemm_a4w4_blockscale(A_q, B_q, A_scale, B_scale, out)
    print(f"Try2 (A_q,B_q,A_s,B_s,out): {r.shape}")
except Exception as e:
    print(f"Try2: {e}")

# Try 3: with shuffled A_scale
try:
    r = gemm_a4w4_blockscale(A_q, B_q, A_scale_sh, B_scale, out)
    print(f"Try3 (A_q,B_q,A_s_sh,B_s,out): {r.shape}")
except Exception as e:
    print(f"Try3: {e}")

# Try 4: A_q as (M, K//2) -> but it's already that
try:
    r = gemm_a4w4_blockscale(A_q, B_q, A_scale_sh, B_scale)
    print(f"Try4 (no out): {r.shape}")
except Exception as e:
    print(f"Try4: {e}")

# Try 5: bf16 A directly
try:
    r = gemm_a4w4_blockscale(A, B_q, A_scale, B_scale, out)
    print(f"Try5 (bf16 A): {r.shape}")
except Exception as e:
    print(f"Try5: {e}")

# Try dynamic_per_group_scaled_quant_fp4
print("\n=== dynamic_per_group_scaled_quant_fp4 ===")
from aiter import dynamic_per_group_scaled_quant_fp4
try:
    r = dynamic_per_group_scaled_quant_fp4(A)
    if isinstance(r, tuple):
        for i, t in enumerate(r):
            print(f"  result[{i}]: {t.shape} {t.dtype}")
    else:
        print(f"  result: {r.shape} {r.dtype}")
except Exception as e:
    print(f"Error: {e}")

# Try with group_size
try:
    r = dynamic_per_group_scaled_quant_fp4(A, 32)
    if isinstance(r, tuple):
        for i, t in enumerate(r):
            print(f"  gs32 result[{i}]: {t.shape} {t.dtype}")
except Exception as e:
    print(f"gs32 Error: {e}")

# Check torch.ops.aiter functions
print("\n=== torch.ops.aiter functions ===")
try:
    ops = dir(torch.ops.aiter)
    for op in sorted(ops):
        if 'fp4' in op.lower() or 'blockscale' in op.lower() or 'mxfp' in op.lower():
            print(f"  {op}")
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
