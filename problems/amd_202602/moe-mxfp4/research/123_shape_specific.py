"""
v123: Shape-specific hard-coded optimization for the 7 benchmark shapes.
Instead of general heuristics, detect exact shape and use per-shape tuned params.
This allows maximally tuned block_m, cktile thresholds, and sorting per shape.
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
    fused_moe_2stages, get_inter_dim,
    get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
)

_orig = get_2stage_cfgs.__wrapped__
_silu = ActivationType.Silu
_per_1x32 = QuantType.per_1x32
_fp4x2 = dtypes.fp4x2
_i32 = dtypes.i32
_fp32 = dtypes.fp32

# Shape-specific config: (E, inter_dim, approx_tok_per_exp_range) -> (use_cktile, sk, block_m_override)
# Benchmark shapes:
#   E=257, d=256:  bs=16(tok/exp=0.56), bs=128(4.48), bs=512(17.9)
#   E=33,  d=512:  bs=16(3.9), bs=128(34.9), bs=512(139.6)
#   E=33,  d=2048: bs=512(139.6)

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

    # Shape-specific decisions
    use_cktile = False
    sk = 1
    bm = None  # None = use default from md

    if inter_dim > 1024:
        # E=33 d=2048: dense, default CK FP4 path is optimal
        use_cktile = False
    elif expert > 100:
        # E=257 shapes (d=256)
        if tokens_per_expert < 2:
            # bs=16: very sparse, cktile sk=2, block_m=16
            use_cktile = True
            sk = 2
            bm = 16
        elif tokens_per_expert < 10:
            # bs=128: moderate sparse, cktile sk=2, block_m=16
            use_cktile = True
            sk = 2
            bm = 16
        else:
            # bs=512: moderate dense, default CK FP4
            use_cktile = False
    else:
        # E=33 shapes (d=512)
        if tokens_per_expert < 5:
            # bs=16: sparse, cktile sk=2, block_m=16
            use_cktile = True
            sk = 2
            bm = 16
        elif tokens_per_expert < 40:
            # bs=128: moderate, cktile sk=1, block_m=16
            use_cktile = True
            sk = 1
            bm = 16
        else:
            # bs=512: dense, default CK FP4
            use_cktile = False

    if use_cktile and is_shuffled:
        md.ksplit = 2
        if bm is not None:
            md.block_m = bm
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

    metadata = _patched(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        dtype, _fp4x2, _fp4x2,
        _per_1x32, isG1U1,
        _silu, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)

    # Pre-allocated sorting
    sort_key = (M, E, model_dim, block_size_M)
    if sort_key not in _sorting_bufs:
        device = topk_ids.device
        max_num_tokens_padded = M * topk + E * block_size_M - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size_M - 1) // block_size_M
        _sorting_bufs[sort_key] = (
            torch.empty(max_num_tokens_padded, dtype=_i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=_fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=_i32, device=device),
            torch.empty(2, dtype=_i32, device=device),
            torch.empty((M, model_dim), dtype=dtype, device=device),
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[sort_key]

    _sort_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, E, block_size_M,
        None, None, 0,
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
