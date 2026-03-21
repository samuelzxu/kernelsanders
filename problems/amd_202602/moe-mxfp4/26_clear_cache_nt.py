"""
Clear get_2stage_cfgs cache after monkey-patching to ensure fresh config
with our overrides. Also patch MOEMetadata to force use_non_temporal_load.
"""
import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

# Monkey-patch use_nt
_fm.use_nt = functools.lru_cache(maxsize=2048)(lambda token, topk, e: True)

# Monkey-patch get_block_size_M
_orig_get_block_size_M = _fm.get_block_size_M.__wrapped__
@functools.lru_cache(maxsize=2048)
def _custom_get_block_size_M(token, topk, expert, inter_dim):
    tokens_per_expert = (token * topk) / expert
    if tokens_per_expert < 64:
        return 32
    else:
        return _orig_get_block_size_M(token, topk, expert, inter_dim)
_fm.get_block_size_M = _custom_get_block_size_M

# Clear the lru_cache on get_2stage_cfgs to ensure our patches take effect
_fm.get_2stage_cfgs.cache_clear()
# Also reset the global cfg_2stages to force re-read
_fm.cfg_2stages = None


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
