"""
v129: 1-stage ASM kernel ONLY for E=257 non-cktile shapes (d_expert=256).
v127 showed: test1 (E=257 d=1024) PASSED, test3 (E=65 d=1536) FAILED.
So 1-stage works for large E with small d_expert.
Restrict to: expert > 100 AND inter_dim <= 1024 AND not cktile.
This only affects E=257 bs=512 (the dense E=257 shape, currently 254us).
E=257 sparse shapes keep cktile. E=33 shapes keep 2-stage CK.
Triggers ASM build at import (~31s, fits in 12-min budget).
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_USE_NT'] = '1'

import torch
import functools
import aiter
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe_2stages, fused_moe_1stage, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
)

# Eagerly trigger ASM module build
_1stage_ok = False
try:
    _ = aiter.fmoe_g1u1
    _1stage_ok = True
except Exception:
    _1stage_ok = False

_orig = get_2stage_cfgs.__wrapped__
_silu = ActivationType.Silu
_per_1x32 = QuantType.per_1x32
_fp4x2 = dtypes.fp4x2
_i32 = dtypes.i32
_fp32 = dtypes.fp32

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

    if inter_dim > 1024:
        use_cktile = False
    elif tokens_per_expert < 5:
        use_cktile = True
        sk = 2
    elif tokens_per_expert < 40 and expert <= 33:
        use_cktile = True
        sk = 1

    if use_cktile and is_shuffled:
        md.ksplit = 2
        md.block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        md.stage1 = functools.partial(
            cktile_moe_stage1,
            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
            k_pad_zeros=hidden_pad // 128 * 128,
            activation=_silu,
            split_k=sk,
        )
        md.stage2 = functools.partial(
            cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=_silu,
        )

    return md

_fm.get_2stage_cfgs = _patched

_sorting_bufs = {}
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd


def _do_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_size_M):
    M, topk = topk_ids.shape
    key = (M, E, model_dim, block_size_M)
    if key not in _sorting_bufs:
        device = topk_ids.device
        max_num_tokens_padded = M * topk + E * block_size_M - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size_M - 1) // block_size_M
        _sorting_bufs[key] = (
            torch.empty(max_num_tokens_padded, dtype=_i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=_fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=_i32, device=device),
            torch.empty(2, dtype=_i32, device=device),
            torch.empty((M, model_dim), dtype=dtype, device=device),
        )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[key]
    _sort_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, E, block_size_M,
        None, None, 0,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, _, _,
        _, _,
        w1, w2,
        w1_scale, w2_scale,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Decide path: 1-stage vs cktile vs 2-stage CK
    tokens_per_expert = (M * topk) / E
    can_cktile = (inter_dim <= 1024 and
                  (tokens_per_expert < 5 or (tokens_per_expert < 40 and E <= 33)))

    # 1-stage: ONLY for E>100 (E=257), d_expert<=1024, non-cktile, ASM available
    use_1stage = (_1stage_ok and isG1U1 and E > 100
                  and inter_dim <= 1024 and not can_cktile)

    if use_1stage:
        # 1-stage fused kernel: quant + GEMM1 + activation + GEMM2 in one path
        block_size_M = 32  # ASM kernel uses 32x512 tile
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _do_sorting(
            topk_ids, topk_weights, E, model_dim, dtype, block_size_M,
        )
        fused_moe_1stage(
            hidden_states, w1, w2,
            topk, sorted_ids, sorted_weights,
            sorted_expert_ids, num_valid_ids,
            moe_buf, isG1U1, block_size_M,
            activation=_silu,
            quant_type=_per_1x32,
            kernelName='',
            q_dtype_a=_fp4x2, q_dtype_w=_fp4x2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None, a2_scale=None,
            num_local_tokens=None,
            M=M, device=topk_ids.device,
            doweight_stage1=False,
        )
    else:
        # 2-stage: cktile for sparse, CK for dense
        metadata = _patched(
            get_padded_M(M), model_dim, inter_dim, E, topk,
            dtype, _fp4x2, _fp4x2,
            _per_1x32, isG1U1,
            _silu, False,
            hidden_pad, intermediate_pad, is_shuffled,
        )
        block_size_M = int(metadata.block_m)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _do_sorting(
            topk_ids, topk_weights, E, model_dim, dtype, block_size_M,
        )
        fused_moe_2stages(
            hidden_states, w1, w2,
            topk, sorted_ids, sorted_weights,
            sorted_expert_ids, num_valid_ids,
            moe_buf, isG1U1, block_size_M,
            activation=_silu,
            quant_type=_per_1x32,
            doweight_stage1=False,
            q_dtype_a=_fp4x2, q_dtype_w=_fp4x2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    return moe_buf
