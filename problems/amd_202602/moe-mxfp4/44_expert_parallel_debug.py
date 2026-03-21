"""
Debug expert-parallel: print raw weight/scale shapes to understand the format.
Use fused_moe as fallback for actual computation.
"""
import torch
import sys
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_printed = False

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _printed
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    if not _printed:
        _printed = True
        M = hidden_states.shape[0]
        E = gate_up_weight.shape[0]
        print(f"[SHAPES] hidden_states: {hidden_states.shape} {hidden_states.dtype}", file=sys.stderr)
        print(f"[SHAPES] gate_up_weight (raw): {gate_up_weight.shape} {gate_up_weight.dtype}", file=sys.stderr)
        print(f"[SHAPES] down_weight (raw): {down_weight.shape} {down_weight.dtype}", file=sys.stderr)
        print(f"[SHAPES] gate_up_weight_scale (raw): {gate_up_weight_scale.shape} {gate_up_weight_scale.dtype}", file=sys.stderr)
        print(f"[SHAPES] down_weight_scale (raw): {down_weight_scale.shape} {down_weight_scale.dtype}", file=sys.stderr)
        print(f"[SHAPES] gate_up_weight_shuffled: {gate_up_weight_shuffled.shape} {gate_up_weight_shuffled.dtype}", file=sys.stderr)
        print(f"[SHAPES] gate_up_weight_scale_shuffled: {gate_up_weight_scale_shuffled.shape} {gate_up_weight_scale_shuffled.dtype}", file=sys.stderr)
        print(f"[SHAPES] config: {config}", file=sys.stderr)
        # Check per-expert slicing
        print(f"[SHAPES] gate_up_weight[0]: {gate_up_weight[0].shape}", file=sys.stderr)
        print(f"[SHAPES] gate_up_weight_scale[0]: {gate_up_weight_scale[0].shape} ndim={gate_up_weight_scale[0].ndim}", file=sys.stderr)
        print(f"[SHAPES] down_weight_scale[0]: {down_weight_scale[0].shape} ndim={down_weight_scale[0].ndim}", file=sys.stderr)

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
