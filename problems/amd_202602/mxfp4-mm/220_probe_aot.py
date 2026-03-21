"""
MXFP4-MM: #220 - Probe what AOT metadata exists on the runner.

This submission prints the AOT directory contents and tries to use
gemm_afp4wfp4_preshuffle with use_aot=True for our shapes to see
if any AOT kernels exist or if we can create the metadata dirs at runtime.
"""
import os, sys, json
import torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

# Print AOT directory contents
aot_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm/aot"
print(f"[PROBE] AOT dir: {aot_dir}", file=sys.stderr)
if os.path.exists(aot_dir):
    dirs = sorted(os.listdir(aot_dir))
    print(f"[PROBE] Found {len(dirs)} AOT dirs:", file=sys.stderr)
    for d in dirs[:30]:
        print(f"  {d}", file=sys.stderr)
        # Check contents
        full = os.path.join(aot_dir, d)
        if os.path.isdir(full):
            files = os.listdir(full)
            for f in files:
                fpath = os.path.join(full, f)
                print(f"    {f} ({os.path.getsize(fpath)} bytes)", file=sys.stderr)
else:
    print(f"[PROBE] AOT dir does not exist!", file=sys.stderr)

# Also check AITER_TRITON_CONFIGS_PATH
print(f"\n[PROBE] Config path: {AITER_TRITON_CONFIGS_PATH}", file=sys.stderr)
gemm_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
if os.path.exists(gemm_dir):
    print(f"[PROBE] Gemm config files:", file=sys.stderr)
    for f in sorted(os.listdir(gemm_dir))[:20]:
        print(f"  {f}", file=sys.stderr)

# Print Triton cache dir
triton_cache = os.path.expanduser("~/.triton/cache")
print(f"\n[PROBE] Triton cache: {triton_cache}", file=sys.stderr)
if os.path.exists(triton_cache):
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(triton_cache) for f in filenames)
    print(f"[PROBE] Cache size: {total // 1024}KB", file=sys.stderr)

# Inject our configs
_PS_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _PS_CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

_inject_configs()

# Standard preshuffle kernel (same as #211)
_b_cache_key = None
_b_cache_w = None
_b_cache_ws = None


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _b_cache_key, _b_cache_w, _b_cache_ws
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key == _b_cache_key:
        B_w = _b_cache_w
        B_ws = _b_cache_ws
    else:
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        B_ws = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
        _b_cache_key = key
        _b_cache_w = B_w
        _b_cache_ws = B_ws

    return gemm_a16wfp4_preshuffle(A, B_w, B_ws, prequant=True, dtype=torch.bfloat16)
