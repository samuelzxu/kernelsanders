"""
v73: Wider selective ksplit - extend to token<=256 and also try E=257.
v72 showed cktile helps for E=33 token<=128. What about wider range?

From v71 analysis, cktile is faster when tokens/expert < ~35.
- E=33 bs=16: 4.4 tok/exp → 33% faster (confirmed)
- E=33 bs=128: 35 tok/exp → 13% faster (confirmed)
- E=33 bs=512: 139 tok/exp → 19% slower
- E=257 bs=16: 0.6 tok/exp → should also benefit!

Try: ksplit=2 for ANY shape where estimated tokens/expert < 64.
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'

import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

_orig_get_ksplit = _fm.get_ksplit

def _patched_get_ksplit(token, topk, expert, inter_dim, model_dim):
    # Estimate tokens per expert
    tokens_per_expert = (token * topk) / expert
    # Use cktile for decode-like scenarios (few tokens per expert)
    # AND only for small inter_dim (large inter_dim has 2x overhead from split_k)
    if tokens_per_expert < 64 and inter_dim <= 1024:
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
