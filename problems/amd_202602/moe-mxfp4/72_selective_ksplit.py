"""
v72: Selective ksplit - use cktile(ksplit=2) for small E=33 shapes,
standard CK for everything else.
v71 showed: cktile is 33% faster for bs=16 E=33 but 107% slower for bs=512 d=2048.
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'

import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

# Override get_ksplit to force ksplit=2 for small E=33 shapes
_orig_get_ksplit = _fm.get_ksplit

def _patched_get_ksplit(token, topk, expert, inter_dim, model_dim):
    # For small E=33 shapes (token <= 128, d_expert <= 512):
    # cktile with ksplit=2 is 33% faster
    if expert == 33 and token <= 128 and inter_dim <= 1024:
        return 2
    return _orig_get_ksplit(token, topk, expert, inter_dim, model_dim)

_fm.get_ksplit = _patched_get_ksplit


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

    return _fm.fused_moe(
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
