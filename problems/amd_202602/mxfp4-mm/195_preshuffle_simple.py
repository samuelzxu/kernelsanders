"""
MXFP4-MM: #195 - gemm_a16wfp4_preshuffle with direct B_scale_sh reshape.

The simplest possible approach:
  gemm_a16wfp4_preshuffle(A_bf16, B_shuffle_reshaped, B_scale_sh_reshaped)

B_scale_sh from task is (N_padded, K//32) in e8m0_shuffle format.
Preshuffle kernel wants (N//32, K) in shuffle_scales format.
These are the SAME data — just a reshape: B_scale_sh[:n].reshape(n//32, k)

B_shuffle from task is (N, K//2). Preshuffle wants (N//16, K*8).
Reshape: B_shuffle.reshape(n//16, k//2 * 16)

Single kernel. No dynamic_mxfp4_quant. No e8m0_shuffle. No Python overhead.
"""
import os, json
import torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

# Config keys use K_LOGICAL per line 421: get_gemm_config(name, M, N, 2*K)
_CONFIGS = {
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
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 16, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
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
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass


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
        # B_shuffle: (N, K//2) → (N//16, K//2*16) = (N//16, K*8)
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)

        # B_scale_sh: (N_padded, K//32) → slice to (N, K//32) → reshape to (N//32, K)
        # e8m0_shuffle format IS shuffle_scales format — just different shape!
        B_scale_u8 = B_scale_sh.view(torch.uint8)
        B_ws = B_scale_u8[:n, :].contiguous().reshape(n // 32, k)

        _b_cache_key = key
        _b_cache_w = B_w
        _b_cache_ws = B_ws

    return gemm_a16wfp4_preshuffle(
        A, B_w, B_ws,
        prequant=True,
        dtype=torch.bfloat16,
    )
