"""
v75: Tuned ksplit threshold based on v74 results.
v74 showed E=257 bs=512 (17 tok/exp) is 7% SLOWER with ksplit=2.
Need tighter threshold: tokens_per_expert < 10 for E=257.

But E=33 bs=128 (35 tok/exp) is 13% FASTER - so threshold depends on expert count.
With fewer experts (E=33), each expert gets more work per token → split_k helps more.
With many experts (E=257), work per expert is tiny → split_k overhead dominates.

Heuristic: tokens_per_expert * inter_dim < threshold
- E=257 d=256: 0.6*256=154 → ksplit=2 (confirmed faster)
- E=257 bs=128 d=256: 4.5*256=1152 → ksplit=2 (confirmed faster)
- E=257 bs=512 d=256: 17*256=4352 → ksplit=0 (confirmed slower with ksplit)
- E=33 bs=16 d=512: 4.4*512=2253 → ksplit=2 (confirmed faster)
- E=33 bs=128 d=512: 35*512=17920 → ksplit=2 (confirmed faster)
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_BYPASS_TUNE_CONFIG'] = '1'

import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

def _patched_get_ksplit(token, topk, expert, inter_dim, model_dim):
    tokens_per_expert = (token * topk) / expert
    # Use cktile for:
    # - Very sparse (< 5 tok/exp): always helps regardless of inter_dim
    # - Moderately sparse (5-40 tok/exp): only for small inter_dim
    if inter_dim > 1024:
        return 0  # Large inter_dim: split_k overhead too high
    if tokens_per_expert < 5:
        return 2  # Very sparse: always use split_k
    if tokens_per_expert < 40 and expert <= 33:
        return 2  # E=33 moderate sparse (proven faster in v72)
    return 0

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
