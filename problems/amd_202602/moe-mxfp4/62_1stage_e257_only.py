"""
v62: Force 1-stage ONLY for E=257 shapes (d_expert=256).
Keep 2-stage for E=33 shapes (where 1-stage crashes).

E=257 shapes are 3/7 of benchmark shapes.
The 1-stage kernel works correctly for these (tested in v33, v59).

Uses Triton quant to avoid module_quant build.
Needs module_moe_asm (builds in ~30s if not cached) BUT ALSO
module_moe_ck2stages for E=33 shapes (~105s if not cached).
Total cold build: 25+105+30 = 160s → may timeout on cold runner.
On warm runner: all cached, 0s builds.
"""
import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Replace HIP quant with Triton quant
def _triton_quant(x, scale=None, quant_dtype=None, shuffle=False, num_rows=None, num_rows_factor=1):
    fp4, bs = dynamic_mxfp4_quant(x)
    return fp4.view(dtypes.fp4x2), bs.view(dtypes.fp8_e8m0)

_original_get_quant = _fm.get_quant
def _patched_get_quant(quant_type):
    if quant_type == QuantType.per_1x32:
        return _triton_quant
    return _original_get_quant(quant_type)
_fm.get_quant = _patched_get_quant

# Force 1-stage ONLY for E=257 shapes
_orig = _fm.get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _patched_cfgs(token, model_dim, inter_dim, expert, topk, *rest, **kw):
    md = _orig(token, model_dim, inter_dim, expert, topk, *rest, **kw)
    # Only E=257, inter_dim=256 uses 1-stage
    # These shapes use the 1tg_ps_32x512 kernel which works correctly
    if expert == 257 and inter_dim == 256:
        md.run_1stage = True
        md.stage1 = functools.partial(
            _fm.fused_moe_1stage,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
        )
    return md

_fm.get_2stage_cfgs = _patched_cfgs


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
