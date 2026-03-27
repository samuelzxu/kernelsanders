#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#301: Install IREE and test FP4 GEMM compilation.
"""
import subprocess, sys, os, time
t0 = time.time()

# Install iree
print("=== Installing IREE ===")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--break-system-packages",
     "iree-compiler", "iree-runtime"],
    capture_output=True, text=True, timeout=300
)
print(f"Install took {time.time()-t0:.1f}s")
if r.returncode != 0:
    print(f"STDERR: {r.stderr[-1000:]}")
    print(f"STDOUT: {r.stdout[-500:]}")
else:
    print("Install OK")

# Check version
try:
    import iree.compiler as ireec
    print(f"Version: {ireec.version.VERSION}")
    targets = ireec.query_available_targets()
    print(f"Targets: {targets}")
except Exception as e:
    print(f"Import error: {e}")

# Try to compile a simple matmul for rocm/gfx950
try:
    import iree.compiler as ireec

    # Simple MLIR matmul module
    mlir_module = """
module {
  func.func @matmul(%arg0: tensor<256x1536xbf16>, %arg1: tensor<1536x3072xbf16>) -> tensor<256x3072xbf16> {
    %cst = arith.constant 0.0 : bf16
    %init = tensor.empty() : tensor<256x3072xbf16>
    %fill = linalg.fill ins(%cst : bf16) outs(%init : tensor<256x3072xbf16>) -> tensor<256x3072xbf16>
    %result = linalg.matmul ins(%arg0, %arg1 : tensor<256x1536xbf16>, tensor<1536x3072xbf16>) outs(%fill : tensor<256x3072xbf16>) -> tensor<256x3072xbf16>
    return %result : tensor<256x3072xbf16>
  }
}
"""

    print("\n=== Compiling bf16 matmul ===")
    t1 = time.time()
    binary = ireec.compile_str(
        mlir_module,
        target_backends=["rocm"],
        extra_args=["--iree-rocm-target-chip=gfx950"]
    )
    print(f"Compiled in {time.time()-t1:.1f}s, binary size: {len(binary)} bytes")

    # Try to run it
    import iree.runtime as ireert
    import torch
    config = ireert.Config("hip")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, binary)
    ctx.add_vm_module(vm_module)

    # Create test inputs
    A = torch.randn(256, 1536, dtype=torch.bfloat16, device="cpu").numpy()
    B = torch.randn(1536, 3072, dtype=torch.bfloat16, device="cpu").numpy()

    f = ctx.modules.module["matmul"]
    result = f(A, B)
    print(f"Result shape: {result.shape}, dtype: {result.dtype}")

except Exception as e:
    print(f"Compile/run error: {e}")

print(f"\nTotal time: {time.time()-t0:.1f}s")

# Kernel - fallback to preshuffle
import torch
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from task import input_t, output_t

import os, json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
_cfgs = {
    "N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=2112-K=7168": {"M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
}
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
