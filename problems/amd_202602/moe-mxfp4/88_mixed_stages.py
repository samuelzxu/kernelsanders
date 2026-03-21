"""
v88: Mixed CK stage1 + cktile stage2 for dense shapes.
CK stage1 uses FP4 MFMA (compute optimal) + FP4 quant for input.
cktile stage2 takes bf16 intermediate directly (no requant needed).
This eliminates the intermediate requant step (~10µs).

For sparse shapes: keep cktile for both stages (same as v85).
For dense shapes: CK stage1 + cktile stage2 (HYBRID).
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'

import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe_2stages, moe_sorting, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
    ck_moe_stage1,
)

_orig = get_2stage_cfgs.__wrapped__

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
    use_cktile_full = False
    use_cktile_stage2_only = False
    sk = 1

    if inter_dim > 1024:
        pass  # Keep standard CK for both stages
    elif tokens_per_expert < 5:
        use_cktile_full = True
        sk = 2  # Very sparse: cktile for both stages
    elif tokens_per_expert < 40 and expert <= 33:
        use_cktile_full = True
        sk = 1  # E=33 moderate: cktile for both stages
    elif tokens_per_expert < 200 and inter_dim <= 512:
        # Dense E=33 d=512: try CK stage1 + cktile stage2
        use_cktile_stage2_only = True

    if use_cktile_full and is_shuffled:
        md.ksplit = 2
        md.block_m = 16 if token < 2048 else 32 if token < 16384 else 64
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
    elif use_cktile_stage2_only and is_shuffled:
        # Keep CK stage1 (uses FP4 quant + MFMA), swap only stage2 to cktile
        # Set ksplit=2 to skip intermediate requant (bf16 path)
        md.ksplit = 2
        # Keep md.stage1 as-is (CK stage1 from CSV or default)
        md.stage2 = functools.partial(
            cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu,
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

    M, topk = topk_ids.shape
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    q_dtype_a = dtypes.fp4x2
    q_dtype_w = dtypes.fp4x2
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    metadata = _patched(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        dtype, q_dtype_a, q_dtype_w,
        QuantType.per_1x32, isG1U1,
        ActivationType.Silu, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_size_M,
    )

    fused_moe_2stages(
        hidden_states, w1, w2,
        topk, sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, isG1U1, block_size_M,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=q_dtype_a, q_dtype_w=q_dtype_w,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    return moe_buf
