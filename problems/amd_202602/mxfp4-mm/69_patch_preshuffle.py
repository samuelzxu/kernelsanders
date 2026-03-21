"""
MXFP4-MM: Patch the preshuffle kernel bug at import time.
Fix: add else clause so 'b' is always defined before use.
Then use preshuffle for M<=16 (faster, no double quant).
"""
import json
import os
import sys
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# Inject Triton configs
_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=2112-K=7168": {
        "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 8, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 1024, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-AFP4WFP4-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass

# PATCH: Fix the preshuffle kernel bug by modifying the source file
def _patch_preshuffle_kernel():
    """Fix NameError('b is not defined') in _gemm_afp4wfp4_preshuffle_kernel."""
    import aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 as kernel_mod
    kernel_path = kernel_mod.__file__
    if kernel_path is None:
        return False

    with open(kernel_path, "r") as f:
        source = f.read()

    # Check if already patched
    if "# PATCHED: added else clause" in source:
        return True

    # Find the buggy pattern and fix it
    bug_pattern = """            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)

            b = (
                b.reshape("""

    fix_pattern = """            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:  # PATCHED: added else clause
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)

            b = (
                b.reshape("""

    if bug_pattern in source:
        source = source.replace(bug_pattern, fix_pattern)
        with open(kernel_path, "w") as f:
            f.write(source)
        # Clear Triton JIT cache for this kernel
        try:
            import importlib
            importlib.reload(kernel_mod)
        except Exception:
            pass
        print(f"[PATCH] Fixed preshuffle kernel at {kernel_path}", file=sys.stderr)
        return True
    else:
        print(f"[PATCH] Bug pattern not found in {kernel_path}", file=sys.stderr)
        return False

_patched = False
try:
    _patched = _patch_preshuffle_kernel()
except Exception as e:
    print(f"[PATCH] Failed: {e}", file=sys.stderr)


import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

# Re-import preshuffle after patching
_preshuffle_ok = False
if _patched:
    try:
        import importlib
        import aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 as kmod
        importlib.reload(kmod)
        import aiter.ops.triton.gemm.basic.gemm_afp4wfp4 as wmod
        importlib.reload(wmod)
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
        _preshuffle_ok = True
        print("[PATCH] preshuffle re-imported after patch", file=sys.stderr)
    except Exception as e:
        print(f"[PATCH] Re-import failed: {e}", file=sys.stderr)

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config

MXFP4_QUANT_BLOCK_SIZE = 32
from task import input_t, output_t


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


_cache_key = None
_cache_val = None


def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    A_q, A_scale = dynamic_mxfp4_quant(A)

    # Preshuffle path for M<=16 (faster, no double quant)
    if _preshuffle_ok and m <= 16:
        A_scale_fmt = A_scale[:m, ...].view(torch.uint8)
        B_scale_fmt = B_scale_sh.view(torch.uint8).view(
            B_scale_sh.shape[0] // MXFP4_QUANT_BLOCK_SIZE, -1
        )
        B_fmt = B_shuffle.view(torch.uint8)
        try:
            return gemm_afp4wfp4_preshuffle(
                A_q.view(torch.uint8), B_fmt,
                A_scale_fmt, B_scale_fmt,
                dtype=torch.bfloat16,
            )
        except Exception:
            pass

    # Triton path
    B_q_uint8 = B_q.view(torch.uint8)
    key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
    if key == _cache_key:
        B_scale = _cache_val
    else:
        if k <= 512:
            _, B_scale = dynamic_mxfp4_quant(B)
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        _cache_key = key
        _cache_val = B_scale

    return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
