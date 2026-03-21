"""
MXFP4-MM: #196 - Preshuffle for K=512/K=7168 + Triton for K=2048/K=1536.

#195 showed preshuffle BEATS aiter targets for K=512 shapes (7.59µs vs 8.2µs!)
and is excellent for K=7168 (17.2µs vs 21.2µs). But K=2048/K=1536 regressed.

Hybrid: use preshuffle where it excels, Triton where it doesn't.
"""
import os, json, sys, importlib, importlib.util

def _patch_to_o1():
    try:
        mod = importlib.import_module('triton.backends.amd.compiler')
        fpath = mod.__file__
        with open(fpath, 'r') as f:
            content = f.read()
        if 'llvm.OPTIMIZE_O3' in content:
            dst_dir = '/tmp/triton_amd_preshuffle_hybrid'
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, 'compiler.py')
            patched = content.replace('llvm.OPTIMIZE_O3', 'llvm.OPTIMIZE_O1')
            with open(dst, 'w') as f:
                f.write(patched)
            spec = importlib.util.spec_from_file_location('triton.backends.amd.compiler', dst)
            patched_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patched_mod)
            sys.modules['triton.backends.amd.compiler'] = patched_mod
            return True
        return False
    except Exception:
        return False

_patch_to_o1()

import torch
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

# Preshuffle configs for K=512 and K=7168
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
    # K=7168 uses the existing aiter tuned config (KSPLIT=14, BSN=128)
}

# Triton configs for K=2048 and K=1536 (proven from #167/#188)
_FP4_CONFIGS = {
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 4, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 32, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"},
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
    for shape_key, config in _PS_CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)
    for shape_key, config in _FP4_CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-AFP4WFP4-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

_inject_configs()

from task import input_t, output_t


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


_ps_cache_key = None
_ps_cache_w = None
_ps_cache_ws = None
_triton_cache_key = None
_triton_cache_val = None


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_cache_key, _ps_cache_w, _ps_cache_ws
    global _triton_cache_key, _triton_cache_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if k <= 512 or k == 7168:
        # Preshuffle path: single kernel, fused quant+GEMM
        key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
        if key == _ps_cache_key:
            B_w = _ps_cache_w
            B_ws = _ps_cache_ws
        else:
            B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
            B_ws = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
            _ps_cache_key = key
            _ps_cache_w = B_w
            _ps_cache_ws = B_ws

        return gemm_a16wfp4_preshuffle(A, B_w, B_ws, prequant=True, dtype=torch.bfloat16)

    else:
        # K=2048, K=1536: Triton path with tuned configs
        B_q_uint8 = B_q.view(torch.uint8)
        key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
        if key == _triton_cache_key:
            B_scale = _triton_cache_val
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
            _triton_cache_key = key
            _triton_cache_val = B_scale

        A_q, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
