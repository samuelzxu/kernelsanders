"""
Tune block_m and use_nt for E=33 shapes that have no CSV config.
The default heuristic may not be optimal. Try specific configs.
"""
import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

_orig = _fm.get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _tuned(token, model_dim, inter_dim, expert, topk, *rest, **kw):
    md = _orig(token, model_dim, inter_dim, expert, topk, *rest, **kw)

    # For E=33 shapes: override block_m to 32 (smallest valid)
    # Smaller block_m = less padding waste = better utilization for sparse routing
    if expert == 33 and md.block_m != 32:
        md.block_m = 32
        md.use_nt = True  # Non-temporal loads help for small per-expert M

    return md

_fm.get_2stage_cfgs = _tuned


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
