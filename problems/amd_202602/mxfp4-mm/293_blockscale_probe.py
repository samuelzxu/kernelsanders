#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#293: Probe gemm_a4w4_blockscale and fp4_utils. Also try gemm_a4w4 with
different ASM kernel names for K=1536 to see if 96x256 tile helps.
Also probe dynamic_per_group_scaled_quant_fp4.
"""
import torch, sys, time
from task import input_t, output_t

# Probe blockscale
print("=== gemm_a4w4_blockscale ===")
try:
    from aiter import gemm_a4w4_blockscale
    import inspect
    sig = inspect.signature(gemm_a4w4_blockscale)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Error: {e}")

# Probe fp4_utils
print("\n=== fp4_utils ===")
try:
    from aiter import fp4_utils
    print(dir(fp4_utils))
except Exception as e:
    print(f"Error: {e}")

# Probe dynamic_per_group_scaled_quant_fp4
print("\n=== dynamic_per_group_scaled_quant_fp4 ===")
try:
    from aiter import dynamic_per_group_scaled_quant_fp4
    import inspect
    sig = inspect.signature(dynamic_per_group_scaled_quant_fp4)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Error: {e}")

# Probe gemm_a4w4_blockscale_tune
print("\n=== gemm_a4w4_blockscale_tune ===")
try:
    from aiter import gemm_a4w4_blockscale_tune
    import inspect
    sig = inspect.signature(gemm_a4w4_blockscale_tune)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Error: {e}")

# Try gemm_a4w4 with explicit kernel name for K=1536
print("\n=== gemm_a4w4_asm with 96x256 ===")
try:
    from aiter import gemm_a4w4_asm, dynamic_mxfp4_quant
    from aiter.ops.triton.quant import e8m0_shuffle
    # Create test data
    M, N, K = 256, 3072, 1536
    A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    A_q, A_scale = dynamic_mxfp4_quant(A)
    # Create dummy B in preshuffle format
    B_q = torch.zeros((N, K//2), dtype=torch.uint8, device="cuda")
    B_scale = torch.zeros((N, K//32), dtype=torch.uint8, device="cuda")
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    A_scale_sh = e8m0_shuffle(A_scale).contiguous()

    # Try calling with kernel name
    try:
        result = gemm_a4w4_asm(A_q, B_q, A_scale_sh, B_scale, out,
                               "f4gemm_bf16_per1x32Fp4_BpreShuffle_96x256",
                               bpreshuffle=True)
        print("96x256 kernel call succeeded!")
    except Exception as e:
        print(f"96x256 error: {e}")

    # Try with 256x256
    try:
        result = gemm_a4w4_asm(A_q, B_q, A_scale_sh, B_scale, out,
                               "f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256",
                               bpreshuffle=True)
        print("256x256 kernel call succeeded!")
    except Exception as e:
        print(f"256x256 error: {e}")

    # Try default (let aiter pick)
    try:
        from aiter import gemm_a4w4
        t0 = time.time()
        for _ in range(10):
            _ = gemm_a4w4(A_q, B_q, A_scale_sh, B_scale, out, bpreshuffle=True)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"gemm_a4w4 default: {(t1-t0)/10*1e6:.1f}µs (Python time, 10 iters)")
    except Exception as e:
        print(f"gemm_a4w4 default error: {e}")

except Exception as e:
    print(f"Setup error: {e}")

# Still need a valid kernel
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
