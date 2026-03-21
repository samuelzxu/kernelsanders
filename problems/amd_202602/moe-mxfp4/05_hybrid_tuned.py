"""
Hybrid approach:
- E=257 shapes: default fused_moe (tuned configs are loaded automatically)
- E=33 shapes: explicit block_size_M=32 (override heuristic 64/128)
Also tries moe_sorting_dispatch_policy=1 for potential sorting improvement.
"""
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]

    # For E=257 shapes, don't override block_m (let tuned config decide)
    # For E=33 shapes, force block_m=32 which we found works better
    if E > 64:
        block_m = None  # Use tuned config
    else:
        block_m = 32  # Override heuristic

    output = fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        block_size_M=block_m,
    )

    return output
