"""
v84: Try cktile sk=1 with block_m=64 for E=33 bs=512 d=512.
v79 used cktile for this shape with block_m=16 → 232µs (10% worse than CK 211µs).
With block_m=64 (matching CK heuristic), the cktile kernel may perform better
for denser scenarios by reducing block overhead.

The key advantage of cktile: 3 kernel launches vs 5 (eliminates quant kernels).
Even if GEMM is slightly slower, fewer launches may compensate at bs=512.
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'

import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm
from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2

_orig = _fm.get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk,
             dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
             activation, doweight_stage1,
             hidden_pad, intermediate_pad, is_shuffled):
    md = _orig(token, model_dim, inter_dim, expert, topk,
               dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
               activation, doweight_stage1,
               hidden_pad, intermediate_pad, is_shuffled)

    tokens_per_expert = (token * topk) / expert
    use_cktile = False
    sk = 1
    bm = None  # Use default if None

    if inter_dim > 1024:
        use_cktile = False
    elif tokens_per_expert < 5:
        use_cktile = True
        sk = 2
        bm = 16  # Sparse: small block_m
    elif expert <= 33 and inter_dim <= 512:
        # ALL E=33 with small inter_dim
        use_cktile = True
        sk = 1
        if tokens_per_expert < 40:
            bm = 16  # Sparse/moderate
        else:
            bm = 64  # Dense: larger block_m

    if use_cktile and is_shuffled:
        md.ksplit = 2
        if bm is None:
            bm = 16 if token < 2048 else 32 if token < 16384 else 64
        md.block_m = bm
        md.stage1 = functools.partial(
            cktile_moe_stage1,
            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
            k_pad_zeros=hidden_pad // 128 * 128,
            activation=ActivationType.Silu,
            split_k=sk,
        )
        md.stage2 = functools.partial(
            cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu,
        )

    return md

_fm.get_2stage_cfgs = _patched

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
