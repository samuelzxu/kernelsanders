"""
v71: Force ksplit=2 to use cktile path with bf16 activations.
This eliminates BOTH quant kernels (input quant + intermediate requant).
Pipeline: moe_sorting → cktile_stage1(bf16) → cktile_stage2(bf16)
Only 3 kernel launches instead of 5.

Combined with HIP_FORCE_DEV_KERNARG=1 for reduced launch overhead.

Risk: cktile path was 2x slower in v25. But v25 didn't have env vars.
The reduced kernel count (3 vs 5) saves ~10µs of quant + ~6µs of launches.
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_KSPLIT'] = '2'  # Force ksplit=2 for cktile path

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


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

    return fused_moe(
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
    )
