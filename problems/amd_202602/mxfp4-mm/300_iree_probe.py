#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#300: Probe IREE availability on runner. Check if iree-compiler
and iree-runtime are installed, and what targets are supported.
"""
import torch, subprocess, sys, os
from task import input_t, output_t

# Probe 1: Check if iree is installed
print("=== IREE availability ===")
try:
    import iree.compiler as ireec
    print(f"iree.compiler version: {ireec.version.VERSION}")
    print(f"Targets: {ireec.query_available_targets()}")
except ImportError:
    print("iree.compiler not installed")
except Exception as e:
    print(f"iree.compiler error: {e}")

try:
    import iree.runtime as ireert
    print(f"iree.runtime available")
except ImportError:
    print("iree.runtime not installed")

# Probe 2: Check pip
print("\n=== pip list ===")
try:
    r = subprocess.run(["pip", "list"], capture_output=True, text=True, timeout=10)
    for line in r.stdout.split('\n'):
        if 'iree' in line.lower() or 'shark' in line.lower() or 'turbine' in line.lower():
            print(line)
    if not any('iree' in l.lower() for l in r.stdout.split('\n')):
        print("No iree packages found in pip list")
except Exception as e:
    print(f"pip error: {e}")

# Probe 3: Check for iree CLI tools
print("\n=== CLI tools ===")
for tool in ['iree-compile', 'iree-run-module', 'iree-benchmark-module']:
    try:
        r = subprocess.run(["which", tool], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(f"{tool}: {r.stdout.strip()}")
        else:
            print(f"{tool}: not found")
    except:
        print(f"{tool}: error")

# Probe 4: Can we pip install iree?
print("\n=== pip install check ===")
try:
    r = subprocess.run(
        ["pip", "install", "--dry-run", "iree-compiler", "iree-runtime"],
        capture_output=True, text=True, timeout=30
    )
    print(r.stdout[-500:] if r.stdout else "no stdout")
    print(r.stderr[-500:] if r.stderr else "no stderr")
except Exception as e:
    print(f"Error: {e}")

# Probe 5: Check ROCm/HIP availability for IREE
print("\n=== ROCm check ===")
try:
    r = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10)
    print(r.stdout[:500])
except: print("rocm-smi not available or error")

# Probe 6: Check if torch can export to MLIR
print("\n=== torch.export check ===")
try:
    import torch.export
    print("torch.export available")
except: print("torch.export not available")

try:
    import torch._dynamo
    print("torch._dynamo available")
except: print("torch._dynamo not available")

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
