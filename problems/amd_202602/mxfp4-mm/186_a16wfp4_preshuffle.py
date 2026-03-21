"""
MXFP4-MM: #186 - Use gemm_a16wfp4_preshuffle for ALL shapes.

aiter ships gemm_a16wfp4_preshuffle which takes BF16 A directly, preshuffled
FP4 B weights, and shuffled B scales. It does quant INLINE in the kernel.

This is a SINGLE kernel launch for the entire pipeline:
  BF16 A + FP4 B_shuffle + E8M0 B_scale_sh → BF16 C

No separate dynamic_mxfp4_quant(A). No e8m0_unshuffle. No separate GEMM launch.
Eliminates 1-2 kernel launches per call vs our current approach.

The function signature:
  gemm_a16wfp4_preshuffle(x=A_bf16, w=B_shuffle_reshaped, w_scales=B_scale_sh)

B_shuffle needs reshaping from (N, K//2) to (N//16, K*8) for the preshuffle kernel.
B_scale_sh can be passed directly (already shuffled).
"""
import os, json
import torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

# Inject per-shape configs for A16WFP4_PRESHUFFLED
_A16W_CONFIGS = {
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
        "M_LEQ_64": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 2, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
}

def _inject_a16w_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _A16W_CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

try:
    _inject_a16w_configs()
except Exception:
    pass


# Pre-cache reshaped B tensors
_b_cache_key = None
_b_cache_w = None
_b_cache_ws = None


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _b_cache_key, _b_cache_w, _b_cache_ws
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache the reshaped B tensors (B_shuffle and B_scale_sh are constant weights)
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key == _b_cache_key:
        B_w = _b_cache_w
        B_ws = _b_cache_ws
    else:
        # Reshape B_shuffle from (N, K//2) to (N//16, K//2 * 16) = (N//16, K*8)
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        # B_scale_sh is in fp8_e8m0 dtype — view as uint8 to avoid KeyError
        B_ws = B_scale_sh.view(torch.uint8)
        _b_cache_key = key
        _b_cache_w = B_w
        _b_cache_ws = B_ws

    return gemm_a16wfp4_preshuffle(
        A,           # BF16 input — quant happens inline in the kernel
        B_w,         # Preshuffled FP4 weights (N//16, K*8)
        B_ws,        # Shuffled E8M0 scales
        prequant=True,
        dtype=torch.bfloat16,
    )
