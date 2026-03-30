"""
Selective block_m tuning: only override for E=33 with small inter_dim.
v54 showed block_m=32 for d=2048 was 21% SLOWER - must exclude large inter_dim.
Also try 4-threadgroup (256x) stage1 kernel for E=33 bs=128.
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

    # Only tune E=33 shapes with small inter_dim (d_expert <= 512)
    if expert == 33 and inter_dim <= 1024:
        # For bs=128 (~34 tokens/expert): try 4-threadgroup stage1
        if 64 <= token <= 256:
            md.block_m = 32
            md.kernelName1 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
            md.use_nt = True

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
