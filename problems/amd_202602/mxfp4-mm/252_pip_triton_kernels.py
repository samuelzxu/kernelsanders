#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#252: Install triton-kernels via pip and probe its matmul capability.
"""
import subprocess, sys, os, time, json, torch

# Time the install
t0 = time.time()
try:
    # Check if already installed
    import importlib.util
    if importlib.util.find_spec("triton_kernels") is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
            "triton-kernels",
        ], timeout=60)
        print(f"[252] pip install took {time.time()-t0:.1f}s", file=sys.stderr)
    else:
        print("[252] triton_kernels already installed", file=sys.stderr)
except Exception as e:
    print(f"[252] install failed: {e}", file=sys.stderr)

# Probe what's available
try:
    import triton_kernels
    print(f"[252] version: {getattr(triton_kernels, '__version__', 'unknown')}", file=sys.stderr)
    print(f"[252] dir: {[x for x in dir(triton_kernels) if not x.startswith('_')]}", file=sys.stderr)
except ImportError as e:
    print(f"[252] import failed: {e}", file=sys.stderr)

try:
    from triton_kernels import matmul
    print(f"[252] matmul found: {type(matmul)}", file=sys.stderr)
    print(f"[252] matmul dir: {[x for x in dir(matmul) if not x.startswith('_')]}", file=sys.stderr)
except Exception as e:
    print(f"[252] matmul import: {e}", file=sys.stderr)

try:
    from triton_kernels import matmul_ogs
    print(f"[252] matmul_ogs: {type(matmul_ogs)}", file=sys.stderr)
except Exception as e:
    print(f"[252] matmul_ogs: {e}", file=sys.stderr)

# Try a minimal matmul
try:
    from triton_kernels.matmul import matmul as tk_matmul
    a = torch.randn(32, 64, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(64, 32, dtype=torch.bfloat16, device='cuda')
    c = tk_matmul(a, b)
    ref = a @ b
    err = (c - ref).abs().max().item()
    print(f"[252] basic matmul works! max_err={err:.6f}", file=sys.stderr)
except Exception as e:
    print(f"[252] basic matmul failed: {e}", file=sys.stderr)

# Preshuffle fallback for correctness
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

_CONFIGS = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"; os.makedirs(cd, exist_ok=True)
    for sk, cfg in _CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f: json.dump(cfg, f)
try: _inject()
except: pass

_bc = [None, None, None]
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
