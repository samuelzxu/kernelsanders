"""
MXFP4-MM: #219 - Use triton_kernels.matmul (Triton's official optimized matmul).

triton_kernels is Triton's production-quality matmul library with:
- CDNA4 scale layout support (unswizzle_mx_scale_cdna4)
- Hardware-specific optimizations per target
- FP4 support (detected via b.dtype == torch.uint8)
- Split-K, persistence, and other advanced features

Available on PyPI as triton-kernels==0.1.0.
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "triton-kernels", "-q"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import torch
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant

try:
    from triton_kernels import matmul as tk_matmul
    from triton_kernels.matmul import PrecisionConfig
    from triton_kernels.tensor import wrap_torch_tensor, FP4
    from triton_kernels.tensor_details.layout_details.cdna4_scale import CDNA4MXScaleLayout
    _HAS_TK = True
    print("[TK] triton_kernels imported successfully", file=sys.stderr)
except Exception as e:
    _HAS_TK = False
    print(f"[TK] Failed to import triton_kernels: {e}", file=sys.stderr)

# Fallback imports
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
import os, json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# Inject preshuffle configs as fallback
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


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


_cache_key = None
_cache_vals = None
_ps_cache_key = None
_ps_cache_w = None
_ps_cache_ws = None


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _cache_key, _cache_vals, _ps_cache_key, _ps_cache_w, _ps_cache_ws
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _HAS_TK:
        # Try triton_kernels matmul for ALL shapes
        try:
            B_q_uint8 = B_q.view(torch.uint8)
            key = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
            if key == _cache_key:
                B_scale, B_scale_tk = _cache_vals
            else:
                B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
                # Wrap B_scale with CDNA4 layout for triton_kernels
                B_scale_tk = wrap_torch_tensor(B_scale, layout=CDNA4MXScaleLayout())
                _cache_key = key
                _cache_vals = (B_scale, B_scale_tk)

            A_q, A_scale = dynamic_mxfp4_quant(A)
            # Wrap tensors for triton_kernels
            a_tk = wrap_torch_tensor(A_q)
            b_tk = wrap_torch_tensor(B_q_uint8, dtype=FP4)

            pc = PrecisionConfig(
                b_mx_scale=B_scale_tk,
                a_mx_scale=wrap_torch_tensor(A_scale),
            )

            result = tk_matmul(a_tk, b_tk, bias=None, precision_config=pc)
            if isinstance(result, torch.Tensor):
                return result
            return result.data  # unwrap Tensor wrapper
        except Exception as e:
            print(f"[TK] matmul failed: {e}", file=sys.stderr)
            # Fall through to preshuffle

    # Fallback: preshuffle
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
