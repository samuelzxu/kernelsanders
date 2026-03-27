"""
v151: Use Swiglu + cktile for ALL shapes (built-in AITER path).
When activation=Swiglu and q_dtype_a=bf16 with per_1x32, AITER's get_2stage_cfgs
returns cktile metadata that skips quantization entirely.
This eliminates quant + requant steps for all shapes.
"""
import os, sys, stat

_JIT_DIR = "/home/runner/aiter/aiter/jit"
_BASE_URL = "https://github.com/samuelzxu/aiter-precompiled/releases/download/v0.3-rocm71"
_MODULES = [
    "module_aiter_enum.so",
    "module_moe_sorting_opus.so",
    "module_moe_sorting.so",
    "module_quant.so",
    "module_activation.so",
    "module_moe_cktile2stages.so",
    "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so",
]

def _install_precompiled():
    import urllib.request
    os.makedirs(_JIT_DIR, exist_ok=True)
    for name in _MODULES:
        path = os.path.join(_JIT_DIR, name)
        if not os.path.exists(path):
            url = f"{_BASE_URL}/{name}"
            try:
                urllib.request.urlretrieve(url, path)
                os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            except Exception:
                pass

try:
    _install_precompiled()
except Exception:
    pass

import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_USE_NT'] = '1'

import torch
import functools
import aiter
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe_2stages, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
)

_sorting_bufs = {}
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd

def _fast_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_size_M):
    M, topk = topk_ids.shape
    key = (M, E, model_dim, block_size_M)
    if key not in _sorting_bufs:
        device = topk_ids.device
        max_num_tokens_padded = M * topk + E * block_size_M - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size_M - 1) // block_size_M
        _sorting_bufs[key] = (
            torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
            torch.empty(2, dtype=dtypes.i32, device=device),
            torch.empty((M, model_dim), dtype=dtype, device=device),
        )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[key]
    _sort_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, E, int(block_size_M),
        None, None, 0,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M, topk = topk_ids.shape
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Use Swiglu activation — triggers cktile BF16 path (no quant)
    metadata = get_2stage_cfgs(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        dtype, dtypes.bf16, dtypes.fp4x2,
        QuantType.per_1x32, isG1U1,
        ActivationType.Swiglu, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fast_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_size_M,
    )

    fused_moe_2stages(
        hidden_states, w1, w2,
        topk, sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, isG1U1, block_size_M,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=dtypes.bf16, q_dtype_w=dtypes.fp4x2,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    return moe_buf
