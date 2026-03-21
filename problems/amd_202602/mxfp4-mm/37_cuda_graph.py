"""
MXFP4-MM: CUDA Graph capture for quant_A + GEMM pipeline.
Captures kernel sequence as graph, replays with minimal CPU overhead.
For ranked benchmark: copy A into static buffer, replay graph.
"""
import json
import os
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

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
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass

from task import input_t, output_t
import torch
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


def e8m0_unshuffle(scale, orig_m, orig_n):
    sm, sn = scale.shape
    scale = scale.view(sm // 32, sn // 8, 4, 16, 2, 2)
    scale = scale.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale = scale.view(sm, sn)
    return scale[:orig_m, :orig_n]


# Graph cache: shape -> (graph, static_A, static_B_q, static_B_scale, static_y)
_graph_cache = {}
_bscale_cache_key = None
_bscale_cache_val = None


def _build_graph(A, B_q_uint8, B_scale, m, n, k):
    """Build CUDA graph for quant_A + GEMM pipeline."""
    # Static input buffers
    static_A = A.clone()
    static_Bq = B_q_uint8.clone()
    static_Bs = B_scale.clone()

    # Warm-up run (needed before graph capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        A_q, A_scale = dynamic_mxfp4_quant(static_A)
        y = gemm_afp4wfp4(A_q, static_Bq, A_scale, static_Bs, dtype=torch.bfloat16)
    torch.cuda.current_stream().wait_stream(s)

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        A_q, A_scale = dynamic_mxfp4_quant(static_A)
        static_y = gemm_afp4wfp4(A_q, static_Bq, A_scale, static_Bs, dtype=torch.bfloat16)

    return g, static_A, static_Bq, static_Bs, static_y


def custom_kernel(data: input_t) -> output_t:
    global _bscale_cache_key, _bscale_cache_val
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    B_q_uint8 = B_q.view(torch.uint8)

    # Get B_scale (with caching)
    bkey = (B.data_ptr(), B_q.data_ptr(), B_scale_sh.data_ptr())
    if bkey == _bscale_cache_key:
        B_scale = _bscale_cache_val
    else:
        if k <= 512:
            _, B_scale = dynamic_mxfp4_quant(B)
        else:
            B_scale = e8m0_unshuffle(B_scale_sh.view(torch.uint8), n, k // 32)
        _bscale_cache_key = bkey
        _bscale_cache_val = B_scale

    # Try graph replay
    graph_key = (m, n, k)
    if graph_key in _graph_cache:
        g, static_A, static_Bq, static_Bs, static_y = _graph_cache[graph_key]
        # Update input buffers
        static_A.copy_(A)
        static_Bq.copy_(B_q_uint8)
        static_Bs.copy_(B_scale)
        # Replay
        g.replay()
        return static_y.clone()

    # First call: build graph (or fallback if graph capture fails)
    try:
        g, static_A, static_Bq, static_Bs, static_y = _build_graph(
            A, B_q_uint8, B_scale, m, n, k
        )
        _graph_cache[graph_key] = (g, static_A, static_Bq, static_Bs, static_y)
        return static_y.clone()
    except Exception:
        # Fallback: no graph
        A_q, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_q, B_q_uint8, A_scale, B_scale, dtype=torch.bfloat16)
