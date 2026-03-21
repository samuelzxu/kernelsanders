"""
Monkey-patch get_ksplit to enable split-K for the d_expert=2048 shape.
For bs=512, E=33, d_expert=2048: K=7168 is large, split-K could help.
The default get_ksplit returns 0 because token*topk > expert.

Also includes use_nt=True and block_m optimizations from v23.
"""
import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

# Force use_nt=True
_fm.use_nt = functools.lru_cache(maxsize=2048)(lambda token, topk, e: True)

# Custom block_m for sparse shapes
_orig_get_block_size_M = _fm.get_block_size_M.__wrapped__
@functools.lru_cache(maxsize=2048)
def _custom_get_block_size_M(token, topk, expert, inter_dim):
    tokens_per_expert = (token * topk) / expert
    if tokens_per_expert < 64:
        return 32
    else:
        return _orig_get_block_size_M(token, topk, expert, inter_dim)
_fm.get_block_size_M = _custom_get_block_size_M

# Override get_ksplit: try split-K for large inter_dim with few experts
_orig_get_ksplit = _fm.get_ksplit.__wrapped__
@functools.lru_cache(maxsize=2048)
def _custom_get_ksplit(token, topk, expert, inter_dim, model_dim):
    # For large d_expert (inter_dim >= 2048) with moderate E,
    # try split-K even when token*topk > expert
    if inter_dim >= 2048 and expert <= 64:
        # Check if model_dim is divisible by split factor
        for k in [2, 3, 4]:
            tilek = 256
            if (model_dim % k == 0) and ((model_dim // k) % tilek == 0):
                return k
    return _orig_get_ksplit(token, topk, expert, inter_dim, model_dim)
_fm.get_ksplit = _custom_get_ksplit


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

    output = _fm.fused_moe(
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

    return output
