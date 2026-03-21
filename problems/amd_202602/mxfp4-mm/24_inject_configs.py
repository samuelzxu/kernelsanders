"""
MXFP4-MM: Inject shape-specific tuned configs at import time.
Writes JSON config files so get_gemm_config finds them.
"""
import json
import os
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

# Write shape-specific configs that may be missing from the installation
_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1,
            "num_warps": 4, "num_stages": 2, "waves_per_eu": 3,
            "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"
        },
        "M_LEQ_32": {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1,
            "num_warps": 4, "num_stages": 2, "waves_per_eu": 3,
            "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"
        },
        "M_LEQ_256": {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1,
            "num_warps": 4, "num_stages": 3, "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"
        },
        "any": {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1,
            "num_warps": 4, "num_stages": 3, "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16, "cache_modifier": None
        }
    },
    "N=4096-K=512": {
        "M_LEQ_32": {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1,
            "num_warps": 4, "num_stages": 2, "waves_per_eu": 3,
            "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"
        },
        "any": {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1,
            "num_warps": 4, "num_stages": 3, "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16, "cache_modifier": None
        }
    },
}

def _inject_configs():
    """Write shape-specific config files if they don't exist."""
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-AFP4WFP4-{shape_key}.json"
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass  # Non-fatal

from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def e8m0_unshuffle(scale, orig_m, orig_n):
    """Reverse e8m0_shuffle: inverse permute and unpad."""
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    A_q, A_scale = dynamic_mxfp4_quant(A)

    if k <= 512:
        B_q_raw, B_scale = dynamic_mxfp4_quant(B)
        return gemm_afp4wfp4(A_q, B_q_raw, A_scale, B_scale, dtype=torch.bfloat16)
    else:
        B_q_uint8 = B_q.view(torch.uint8)
        B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
