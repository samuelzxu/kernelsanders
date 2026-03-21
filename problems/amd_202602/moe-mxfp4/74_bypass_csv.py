"""
v74: Bypass CSV tune configs to allow ksplit override for E=257 shapes.
AITER_BYPASS_TUNE_CONFIG=1 forces default heuristics path for all shapes.
Combined with selective ksplit=2 for small batch / many experts scenarios.

Risk: E=257 shapes lose tuned kernel selection (kernelName1/2 = "").
The CK runtime auto-selects which may be slightly different from tuned.
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_BYPASS_TUNE_CONFIG'] = '1'

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

_orig_get_ksplit = _fm.get_ksplit

def _patched_get_ksplit(token, topk, expert, inter_dim, model_dim):
    tokens_per_expert = (token * topk) / expert
    # Use cktile for sparse scenarios (few tokens/expert) with small inter_dim
    if tokens_per_expert < 40 and inter_dim <= 1024:
        return 2
    return 0  # Default: no split

_fm.get_ksplit = _patched_get_ksplit

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
