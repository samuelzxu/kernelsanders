#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#296: Try gemm_a4w4_blockscale. Fix imports.
Also try calling it with the task's B_shuffle and B_scale_sh directly.
"""
import torch, sys, time
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

# Find the right quant imports
print("=== Finding imports ===")
quant_fn = None
e8m0_sh = None
try:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    quant_fn = dynamic_mxfp4_quant
    print("quant from aiter.ops.triton.quant")
except:
    try:
        from aiter.ops.triton._triton_kernels.quant.quant import dynamic_mxfp4_quant
        quant_fn = dynamic_mxfp4_quant
        print("quant from _triton_kernels")
    except:
        print("quant not found in expected places")

# Find e8m0_shuffle
for mod_path in ['aiter.ops.triton.quant', 'aiter.ops.triton._triton_kernels.quant.quant',
                 'aiter.ops.triton.gemm.basic.gemm_a16wfp4']:
    try:
        m = __import__(mod_path, fromlist=['e8m0_shuffle'])
        if hasattr(m, 'e8m0_shuffle'):
            e8m0_sh = m.e8m0_shuffle
            print(f"e8m0_shuffle from {mod_path}")
            break
    except: pass

if e8m0_sh is None:
    # Try aiter top-level
    try:
        import aiter
        if hasattr(aiter, 'e8m0_shuffle'):
            e8m0_sh = aiter.e8m0_shuffle
            print("e8m0_shuffle from aiter")
    except: pass

if e8m0_sh is None:
    print("e8m0_shuffle not found, trying torch.ops.aiter")
    try:
        e8m0_sh = torch.ops.aiter.e8m0_shuffle
        print("e8m0_shuffle from torch.ops.aiter")
    except: print("Not in torch.ops.aiter either")

# List ALL torch.ops.aiter functions
print("\n=== ALL torch.ops.aiter ===")
try:
    import aiter  # trigger module builds
    ops = sorted([x for x in dir(torch.ops.aiter) if not x.startswith('_')])
    for op in ops:
        print(f"  {op}")
except Exception as e:
    print(f"Error listing ops: {e}")

# Try gemm_a4w4_blockscale directly via torch.ops
print("\n=== gemm_a4w4_blockscale via torch.ops ===")
try:
    M, N, K = 4, 2880, 512
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    if quant_fn:
        A_q, A_scale = quant_fn(A)
        print(f"A_q: {A_q.shape}, A_scale: {A_scale.shape}")
        B_q = torch.zeros((N, K//2), dtype=torch.uint8, device="cuda")
        B_scale = torch.zeros((N, K//32), dtype=torch.uint8, device="cuda")
        out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        # Try different argument patterns
        for args_name, args in [
            ("5args", (A_q, B_q, A_scale, B_scale, out)),
            ("4args", (A_q, B_q, A_scale, B_scale)),
            ("6args+tune", (A_q, B_q, A_scale, B_scale, out, "")),
        ]:
            try:
                r = torch.ops.aiter.gemm_a4w4_blockscale(*args)
                print(f"  {args_name}: OK {r.shape if hasattr(r,'shape') else type(r)}")
            except Exception as e:
                print(f"  {args_name}: {str(e)[:100]}")
except Exception as e:
    print(f"Setup error: {e}")

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
