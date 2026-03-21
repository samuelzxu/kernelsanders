"""
Per-shape block_m optimization:
- E=257: let tuned configs decide (None)
- E=33, M<=16: block_m=32 (sparse, small blocks good)
- E=33, M=128: block_m=32 (tested: 32 beats heuristic 64)
- E=33, M=512, d_expert<=512: block_m=64 (heuristic default, test both)
- E=33, M=512, d_expert=2048: block_m=64 (test vs heuristic 128)
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
    d_expert = config["d_expert"]

    # Shape-aware block_m
    tokens_per_expert = (M * topk) / E
    if E > 64:
        block_m = None  # E=257: use tuned configs from CSV
    elif tokens_per_expert < 64:
        block_m = 32   # Sparse: small blocks reduce padding waste
    else:
        block_m = 64   # Dense: larger blocks, but 64 beats heuristic 128

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
