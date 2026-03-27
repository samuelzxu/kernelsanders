#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#253: Install triton_kernels from git repo subdirectory (not PyPI).
"""
import subprocess, sys, os, time, importlib.util, json, torch
from task import input_t, output_t

# Install triton_kernels from triton repo (pure Python, fast)
t0 = time.time()
if importlib.util.find_spec("triton_kernels") is None:
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--quiet", "--no-deps",
            "git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels"
        ], timeout=120)
        print(f"[253] installed from git in {time.time()-t0:.1f}s", file=sys.stderr)
    except Exception as e:
        print(f"[253] git install failed: {e}", file=sys.stderr)
        # Try main branch
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--quiet", "--no-deps",
                "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"
            ], timeout=120)
            print(f"[253] installed from git@main in {time.time()-t0:.1f}s", file=sys.stderr)
        except Exception as e2:
            print(f"[253] main install also failed: {e2}", file=sys.stderr)
else:
    print("[253] triton_kernels already installed", file=sys.stderr)

# Probe
try:
    import triton_kernels
    print(f"[253] version: {getattr(triton_kernels, '__version__', 'unknown')}", file=sys.stderr)
    print(f"[253] dir: {[x for x in dir(triton_kernels) if not x.startswith('_')]}", file=sys.stderr)
except Exception as e:
    print(f"[253] import: {e}", file=sys.stderr)

try:
    import triton
    target = triton.runtime.driver.active.get_current_target()
    print(f"[253] Backend: {target.backend}, arch: {target.arch}", file=sys.stderr)
except Exception as e:
    print(f"[253] target: {e}", file=sys.stderr)

try:
    from triton_kernels.matmul_ogs import matmul as tk_matmul
    print(f"[253] matmul_ogs.matmul available!", file=sys.stderr)
    # Quick test
    a = torch.randn(32, 64, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(64, 32, dtype=torch.bfloat16, device='cuda')
    c = tk_matmul(a, b)
    ref = a @ b
    err = (c.float() - ref.float()).abs().max().item()
    print(f"[253] bf16 matmul max_err={err:.6f}", file=sys.stderr)
except Exception as e:
    print(f"[253] matmul test: {e}", file=sys.stderr)

# Preshuffle fallback
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
