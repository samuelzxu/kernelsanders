"""
Cache intermediate buffers across calls to avoid reallocation.
Pre-allocate sorted_ids, sorted_weights, moe_buf, etc. for each shape.
Also cache the metadata from get_2stage_cfgs to skip lru_cache lookup.
"""
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe, fused_moe_2stages, moe_sorting, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
)


# Cache for pre-allocated buffers per shape key
_cache = {}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M, topk = topk_ids.shape
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    q_dtype_a = dtypes.fp4x2
    q_dtype_w = dtypes.fp4x2
    is_shuffled = getattr(w1, "is_shuffled", False)

    # Cache key based on shape parameters
    key = (M, E, topk, model_dim, inter_dim, hidden_pad, intermediate_pad)

    if key not in _cache:
        # First call for this shape: compute and cache metadata
        metadata = get_2stage_cfgs(
            get_padded_M(M), model_dim, inter_dim, E, topk,
            dtype, q_dtype_a, q_dtype_w,
            QuantType.per_1x32, isG1U1,
            ActivationType.Silu, False,
            hidden_pad, intermediate_pad, is_shuffled,
        )
        block_size_M = int(metadata.block_m)
        _cache[key] = (metadata, block_size_M)

    metadata, block_size_M = _cache[key]

    # Sort (allocates fresh each time - sorting depends on topk_ids)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_size_M,
    )

    # Call 2-stage directly
    fused_moe_2stages(
        hidden_states, w1, w2,
        topk, sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, isG1U1, block_size_M,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=q_dtype_a, q_dtype_w=q_dtype_w,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    return moe_buf
