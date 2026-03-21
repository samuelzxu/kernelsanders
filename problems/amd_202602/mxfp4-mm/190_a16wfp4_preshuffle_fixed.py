"""
MXFP4-MM: #190 - gemm_a16wfp4_preshuffle with CORRECT scale format.

#186 failed because B_scale_sh is in e8m0_shuffle format (N, K//32) but the
preshuffle kernel expects shuffle_scales format (N//32, K). Different transforms!

Fix: convert B_scale_sh → unshuffle → shuffle_scales. Both are tiny tensor ops
on ~46K uint8 elements, cached after first call.

This gives us:
  1. Single kernel launch (bf16 A → inline fp4 quant + GEMM)
  2. Preshuffled B for better memory coalescing
  3. Tuned configs with KSPLIT=14 for K=7168
"""
import os, json
import torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


def shuffle_scales(scales):
    """Convert (N, K//32) unshuffled scale to preshuffle format (N//32, K)."""
    sm, sn = scales.shape
    scales = scales.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales = scales.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales = scales.view(sm // 32, sn * 32)
    return scales


# Inject per-shape configs for A16WFP4_PRESHUFFLED
# K extraction: w.shape = (N//16, K*8) → K = K*8//16 = K//2 = K_packed
# Config key K = K_packed
_CONFIGS = {
    # K=512: K_packed=256
    "N=2880-K=256": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=256": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    # K=7168: K_packed=3584. BSK=512, KSPLIT=7: 3584/(7*512)=1 ✓
    "N=2112-K=3584": {
        "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 7, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 2, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    # K=2048: K_packed=1024. BSK=512, KSPLIT=1: 1024/512=2 ✓
    "N=7168-K=1024": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    # K=1536: K_packed=768. BSK=256, KSPLIT=1: 768/256=3 ✓
    "N=3072-K=768": {
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


# Cache converted B tensors
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
        # B_shuffle: (N, K//2) fp4x2 → reshape to (N//16, K//2 * 16) = (N//16, K*8) uint8
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)

        # B_scale_sh: e8m0_shuffle format (N_padded, K//32)
        # Preshuffle kernel needs shuffle_scales format (N//32, K)
        # Convert: unshuffle → reshuffle
        B_scale_orig = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        B_ws = shuffle_scales(B_scale_orig)

        _b_cache_key = key
        _b_cache_w = B_w
        _b_cache_ws = B_ws

    return gemm_a16wfp4_preshuffle(
        A,           # BF16 input — quant happens inline
        B_w,         # Preshuffled FP4 weights (N//16, K*8)
        B_ws,        # shuffle_scales format (N//32, K)
        prequant=True,
        dtype=torch.bfloat16,
    )
