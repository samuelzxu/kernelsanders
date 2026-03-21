"""
Force fused_moe to use the 1-stage path by monkey-patching get_2stage_cfgs
to always return run_1stage=True. This lets fused_moe handle all the format
conversions correctly while using the single-kernel assembly path.
"""
import torch
import functools
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

# Patch get_2stage_cfgs to force 1-stage
_orig = _fm.get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _force_1stage(token, model_dim, inter_dim, expert, topk, *rest_args, **kwargs):
    metadata = _orig(token, model_dim, inter_dim, expert, topk, *rest_args, **kwargs)
    # Force 1-stage for inter_dim <= 1536 (d_expert <= 768)
    # The assembly kernel has precision issues for large inter_dim (d_expert=2048)
    if inter_dim <= 1536:
        metadata.run_1stage = True
        metadata.stage1 = functools.partial(
            _fm.fused_moe_1stage,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
        )
    return metadata

_fm.get_2stage_cfgs = _force_1stage


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
