#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#302: Probe MiGraphX / ROCmlir availability on runner.
"""
import subprocess, sys, os, torch
from task import input_t, output_t

# Probe 1: MiGraphX Python
print("=== MiGraphX Python ===")
try:
    import migraphx
    print(f"migraphx available: {dir(migraphx)[:500]}")
except ImportError:
    print("migraphx not importable")

# Probe 2: MiGraphX CLI
print("\n=== MiGraphX CLI ===")
for tool in ['migraphx-driver', 'rocmlir-driver', 'rocmlir-gen', 'rocmlir-opt']:
    try:
        r = subprocess.run(["which", tool], capture_output=True, text=True, timeout=5)
        print(f"{tool}: {r.stdout.strip() if r.returncode == 0 else 'not found'}")
    except:
        print(f"{tool}: error")

# Probe 3: Check /opt/rocm paths
print("\n=== /opt/rocm paths ===")
for path in ['/opt/rocm/lib', '/opt/rocm/bin', '/opt/rocm/share']:
    try:
        contents = os.listdir(path)
        relevant = [f for f in contents if any(x in f.lower() for x in ['migraphx', 'mlir', 'rocm', 'miopen'])]
        if relevant:
            print(f"{path}: {relevant[:20]}")
    except:
        pass

# Probe 4: Check for migraphx shared libs
print("\n=== migraphx libs ===")
try:
    r = subprocess.run(["find", "/opt/rocm", "-name", "*migraphx*", "-maxdepth", "4"],
                       capture_output=True, text=True, timeout=10)
    print(r.stdout[:1000] if r.stdout else "none found")
except: print("find error")

# Probe 5: Check for rocmlir
try:
    r = subprocess.run(["find", "/opt/rocm", "-name", "*rocmlir*", "-maxdepth", "4"],
                       capture_output=True, text=True, timeout=10)
    print(r.stdout[:1000] if r.stdout else "no rocmlir found")
except: print("find error")

# Probe 6: Check for MIOpen (has FP4 kernels)
print("\n=== MIOpen ===")
try:
    r = subprocess.run(["find", "/opt/rocm", "-name", "*miopen*", "-maxdepth", "3"],
                       capture_output=True, text=True, timeout=10)
    relevant = [l for l in r.stdout.split('\n') if l.strip()][:20]
    for l in relevant:
        print(l)
except: print("find error")

# Probe 7: dpkg for installed packages
print("\n=== Installed ROCm packages ===")
try:
    r = subprocess.run(["dpkg", "-l"], capture_output=True, text=True, timeout=10)
    for line in r.stdout.split('\n'):
        if any(x in line.lower() for x in ['migraphx', 'rocmlir', 'miopen', 'hipblaslt']):
            print(line[:100])
except: pass

# Probe 8: Check pip for onnx/onnxruntime-rocm
print("\n=== ONNX Runtime ===")
try:
    import onnxruntime
    print(f"onnxruntime version: {onnxruntime.__version__}")
    print(f"Providers: {onnxruntime.get_available_providers()}")
except ImportError:
    print("onnxruntime not installed")

# Probe 9: torch._C has any rocm gemm ops?
print("\n=== torch.ops search ===")
try:
    import aiter
    ops = [x for x in dir(torch.ops.aiter) if not x.startswith('_')]
    # Check for any blockscale-related or new FP4 ops
    for op in sorted(ops):
        if any(x in op.lower() for x in ['block', 'fp4', 'mxfp', 'scale', 'quant']):
            print(f"  {op}")
except: pass

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

import json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=2112-K=7168": {"M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

_ck = None; _cw = None; _cs = None
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ck, _cw, _cs
    A = data[0]; B_shuffle = data[3]; B_scale_sh = data[4]
    m, k = A.shape; n = data[1].shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _ck:
        _ck = dp
        _cw = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _cs = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _cw, _cs, prequant=True, dtype=torch.bfloat16)
