"""
Hybrid: Use 1-stage fmoe_g1u1 for shapes where it works (d_expert<=512),
2-stage fused_moe for d_expert=2048 (where 1-stage failed correctness).

The 1-stage path does sort+quant+GEMM1+SwiGLU+reQuant+GEMM2+reduction
ALL in one assembly kernel, eliminating sorting and quant overhead.

From profiling: sorting=14%, quant=13% → potential 27% savings.
"""
import torch
import functools
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_inter_dim, BLOCK_SIZE_M
from aiter import get_hip_quant as get_quant
from aiter.utility import fp4_utils


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
    d_expert = config["d_expert"]

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = hidden_states.dtype

    # For d_expert=2048, 1-stage kernel has correctness issues → use 2-stage
    if d_expert > 1024:
        return fused_moe(
            hidden_states, w1, w2, topk_weights, topk_ids,
            expert_mask=None, activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32, doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    # 1-stage path for d_expert <= 1024
    block_m = BLOCK_SIZE_M  # 32

    # Sort tokens
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )

    # Quantize activations to fp4 (shuffle=False for non-shuffled scale format)
    quant_func = get_quant(QuantType.per_1x32)
    a1, a1_scale = quant_func(
        hidden_states, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False,
    )

    # Sort scales for MoE access pattern
    a1_scale = fp4_utils.moe_mxfp4_sort(
        a1_scale, sorted_ids, num_valid_ids, M, block_m,
    )

    # View weight scales as (E, -1) for 1-stage kernel
    w1_scale = gate_up_weight_scale_shuffled.view(E, -1)
    w2_scale = down_weight_scale_shuffled.view(E, -1)

    # Call 1-stage assembly kernel
    aiter.fmoe_g1u1(
        moe_buf,           # output
        a1,                # quantized activations
        w1,                # gate_up weights (shuffled)
        w2,                # down weights (shuffled)
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,          # sorted activation scales
        w1_scale,          # gate_up weight scales
        w2_scale,          # down weight scales
        "",                # kernelName (auto-select)
        fc2_smooth_scale=None,
        activation=ActivationType.Silu,
    )

    return moe_buf
