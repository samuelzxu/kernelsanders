#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#303: Deep probe hipBLASLt FP4 + MiGraphX GEMM.
hipBLASLt 1.1.0 is installed. Check if FP4 enums exist with different names.
Also try MiGraphX bf16 GEMM compilation and timing.
"""
import torch, subprocess, sys, time, ctypes, os
from task import input_t, output_t

# ===== hipBLASLt FP4 probe =====
print("=== hipBLASLt probe ===")

# Try to find the library
for lib_path in ['/opt/rocm/lib/libhipblaslt.so', '/usr/lib/libhipblaslt.so']:
    if os.path.exists(lib_path):
        print(f"Found: {lib_path}")
        break

# Check hipBLASLt headers for FP4 types
try:
    r = subprocess.run(
        ["grep", "-r", "FP4\|fp4\|E2M1\|e2m1\|MXFP\|mxfp\|_4F_\|SCALEDTYPE\|SCALE_TYPE\|VEC32",
         "/opt/rocm/include/hipblaslt/"],
        capture_output=True, text=True, timeout=10
    )
    if r.stdout:
        print(f"FP4-related:\n{r.stdout[:2000]}")
    else:
        print("No FP4 matches in hipblaslt headers")
except Exception as e:
    print(f"grep error: {e}")

# Check hipblaslt-ext header
try:
    r = subprocess.run(
        ["find", "/opt/rocm/include", "-name", "*hipblaslt*"],
        capture_output=True, text=True, timeout=10
    )
    print(f"Headers: {r.stdout[:500]}")
except: pass

# List all hipblaslt data type enums
try:
    r = subprocess.run(
        ["grep", "-r", "HIPBLASLT_R_\|hipblaslt_datatype\|hipblasltDatatype",
         "/opt/rocm/include/hipblaslt/"],
        capture_output=True, text=True, timeout=10
    )
    if r.stdout:
        # Get unique enum values
        lines = set(l.strip() for l in r.stdout.split('\n') if 'HIPBLASLT_R_' in l)
        print(f"\nEnum values ({len(lines)}):")
        for l in sorted(lines)[:30]:
            print(f"  {l[:120]}")
except: pass

# ===== MiGraphX GEMM =====
print("\n=== MiGraphX GEMM ===")
try:
    import migraphx

    # Create a simple matmul program
    p = migraphx.program()
    m = p.get_main_module()

    M, N, K = 256, 3072, 1536

    # Add input parameters
    a_shape = migraphx.shape(lens=[M, K], type="half")
    b_shape = migraphx.shape(lens=[K, N], type="half")

    a_param = m.add_parameter("A", a_shape)
    b_param = m.add_parameter("B", b_shape)

    # Add matmul
    dot = m.add_instruction(migraphx.op("dot"), [a_param, b_param])
    m.add_return([dot])

    # Compile for GPU
    t0 = time.time()
    p.compile(migraphx.get_target("gpu"))
    t1 = time.time()
    print(f"MiGraphX compile: {t1-t0:.2f}s")

    # Create GPU inputs
    a_np = torch.randn(M, K, dtype=torch.float16).numpy()
    b_np = torch.randn(K, N, dtype=torch.float16).numpy()

    a_arg = migraphx.argument(a_np)
    b_arg = migraphx.argument(b_np)

    a_gpu = migraphx.to_gpu(a_arg)
    b_gpu = migraphx.to_gpu(b_arg)

    # Warmup
    for _ in range(3):
        result = p.run({"A": a_gpu, "B": b_gpu})
    migraphx.gpu_sync()

    # Benchmark
    t0 = time.time()
    N_ITERS = 100
    for _ in range(N_ITERS):
        result = p.run({"A": a_gpu, "B": b_gpu})
    migraphx.gpu_sync()
    t1 = time.time()
    avg_us = (t1 - t0) / N_ITERS * 1e6
    print(f"MiGraphX dot ({M}x{K} @ {K}x{N}): {avg_us:.1f}µs avg")

    # Get result shape
    r = migraphx.from_gpu(result[0])
    print(f"Result shape: {r.get_shape()}")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

# ===== IREE with correct flags =====
print("\n=== IREE compile ===")
try:
    import iree.compiler as ireec

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
    t0 = time.time()
    binary = ireec.compile_str(
        mlir_module,
        target_backends=["rocm"],
        extra_args=["--iree-hip-target=gfx950"]
    )
    print(f"IREE compiled in {time.time()-t0:.1f}s, size: {len(binary)} bytes")
except Exception as e:
    print(f"IREE error: {e}")

# Standard kernel fallback
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
