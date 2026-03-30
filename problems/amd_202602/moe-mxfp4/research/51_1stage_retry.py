"""
Retry 1-stage fmoe_g1u1 approach. On warm runners (cached CK modules),
the module_moe_asm JIT build (~30s) should fit within 12-min timeout.

Force 1-stage for inter_dim <= 1024 (d_expert <= 512).
d_expert=2048 uses 2-stage (precision issue with 1-stage).
"""
import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

# Force 1-stage for supported shapes
_orig = _fm.get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk, *rest, **kw):
    md = _orig(token, model_dim, inter_dim, expert, topk, *rest, **kw)
    # 1-stage for inter_dim <= 1024 (d_expert <= 512)
    # d_expert=2048 (inter_dim=2048) has 2-element precision issue
    if inter_dim <= 1024:
        md.run_1stage = True
        md.stage1 = functools.partial(
            _fm.fused_moe_1stage,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
        )
    return md

_fm.get_2stage_cfgs = _patched


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
