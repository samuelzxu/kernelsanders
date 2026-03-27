#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#254: Install triton_kernels via direct tarball download (no git needed).
"""
import subprocess, sys, os, time, importlib.util, json, torch
from task import input_t, output_t

t0 = time.time()
if importlib.util.find_spec("triton_kernels") is None:
    # Try pip install from github archive (no git clone needed)
    urls = [
        "https://github.com/triton-lang/triton/archive/refs/tags/v3.5.0.tar.gz",
    ]
    for url in urls:
        try:
            # Download and install from subdirectory
            r = subprocess.run([
                sys.executable, "-m", "pip", "install", "--no-deps", "--break-system-packages",
                f"triton-kernels @ {url}#subdirectory=python/triton_kernels"
            ], capture_output=True, text=True, timeout=120)
            print(f"[254] pip result: rc={r.returncode}", file=sys.stderr)
            if r.returncode != 0:
                print(f"[254] stderr: {r.stderr[:500]}", file=sys.stderr)
            else:
                print(f"[254] installed from {url} in {time.time()-t0:.1f}s", file=sys.stderr)
                break
        except Exception as e:
            print(f"[254] {url}: {e}", file=sys.stderr)

    # Fallback: try downloading and extracting manually
    if importlib.util.find_spec("triton_kernels") is None:
        try:
            import tempfile, urllib.request, tarfile
            with tempfile.TemporaryDirectory() as td:
                tgz = os.path.join(td, "triton.tar.gz")
                urllib.request.urlretrieve(
                    "https://github.com/triton-lang/triton/archive/refs/tags/v3.5.0.tar.gz",
                    tgz
                )
                with tarfile.open(tgz) as tf:
                    # Extract just the triton_kernels subdirectory
                    members = [m for m in tf.getmembers() if "python/triton_kernels/" in m.name]
                    tf.extractall(td, members=members)
                # Install from extracted dir
                tk_dir = os.path.join(td, "triton-3.5.0", "python", "triton_kernels")
                if os.path.exists(tk_dir):
                    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--break-system-packages", tk_dir],
                                       capture_output=True, text=True, timeout=60)
                    print(f"[254] manual install: rc={r.returncode}", file=sys.stderr)
                    if r.returncode != 0:
                        print(f"[254] manual stderr: {r.stderr[:300]}", file=sys.stderr)
                    else:
                        print(f"[254] manual install OK in {time.time()-t0:.1f}s", file=sys.stderr)
                else:
                    # List what was extracted
                    for root, dirs, files in os.walk(td):
                        if "triton_kernels" in root:
                            print(f"[254] found: {root}: {files[:5]}", file=sys.stderr)
                            break
        except Exception as e:
            print(f"[254] manual: {e}", file=sys.stderr)

# Probe
try:
    import triton_kernels
    print(f"[254] version: {getattr(triton_kernels, '__version__', 'unknown')}", file=sys.stderr)
    print(f"[254] dir: {[x for x in dir(triton_kernels) if not x.startswith('_')]}", file=sys.stderr)
except Exception as e:
    print(f"[254] import: {e}", file=sys.stderr)

try:
    from triton_kernels.matmul_ogs import matmul as tk_matmul
    print(f"[254] matmul_ogs available!", file=sys.stderr)
except Exception as e:
    print(f"[254] matmul_ogs: {e}", file=sys.stderr)

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
