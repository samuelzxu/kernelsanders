#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#292: Probe aiter for any GEMM functions we haven't tried.
List available f4gemm ASM kernels, check for hipBLASLt, etc.
"""
import os, torch, glob
from task import input_t, output_t

# Probe 1: List all f4gemm ASM kernels available
print("=== F4GEMM ASM kernels ===")
for f in sorted(glob.glob("/home/runner/aiter/hsa/gfx950/f4gemm/*.co")):
    print(os.path.basename(f))

# Probe 2: List all gemm-related modules in aiter
print("\n=== aiter.ops.triton.gemm contents ===")
import aiter.ops.triton.gemm as g
for d in sorted(dir(g)):
    if not d.startswith('_'):
        print(d)

print("\n=== aiter.ops.triton.gemm.basic contents ===")
import aiter.ops.triton.gemm.basic as gb
for d in sorted(dir(gb)):
    if not d.startswith('_'):
        print(d)

# Probe 3: Check for any new/different gemm wrappers
print("\n=== aiter top-level gemm functions ===")
import aiter
for name in sorted(dir(aiter)):
    if 'gemm' in name.lower() or 'fp4' in name.lower() or 'mxfp' in name.lower():
        print(name)

# Probe 4: Check aiter.ops contents
print("\n=== aiter.ops contents ===")
import aiter.ops as ops
for d in sorted(dir(ops)):
    if 'gemm' in d.lower() or 'fp4' in d.lower() or 'quant' in d.lower():
        print(d)

# Probe 5: List all .co files (compiled objects)
print("\n=== All .co files ===")
for f in sorted(glob.glob("/home/runner/aiter/hsa/gfx950/**/*.co", recursive=True)):
    print(os.path.basename(f))

# Probe 6: Check hipblaslt availability
print("\n=== hipBLASLt check ===")
try:
    import hipblas
    print("hipblas available:", dir(hipblas))
except: print("hipblas not available")
try:
    from aiter.ops import hipblaslt_gemm
    print("hipblaslt_gemm available")
except: print("hipblaslt_gemm not directly importable")

# Probe 7: Check for any kernel config JSONs already existing
import aiter.ops.triton.utils.core as core
cfg_path = core.AITER_TRITON_CONFIGS_PATH
print(f"\n=== Config path: {cfg_path} ===")
for f in sorted(glob.glob(f"{cfg_path}/gemm/*.json")):
    print(os.path.basename(f))

# Probe 8: Check aiter version
print(f"\n=== aiter version ===")
try: print(aiter.__version__)
except: print("no __version__")

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
