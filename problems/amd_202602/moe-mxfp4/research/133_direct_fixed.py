"""
v133: Fixed direct pipeline - skip quant for cktile shapes (ksplit > 1).
v132 failed because it called fused_dynamic_mxfp4_quant_moe_sort with block_m=16
for cktile shapes. But fused_moe_2stages SKIPS quant entirely for cktile
(line 1083: q_dtype_a==fp4x2 and ksplit>1 → a1=hidden_states, a1_scale=None).
This version correctly handles both paths.
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
    get_inter_dim, get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
)
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

_orig = get_2stage_cfgs.__wrapped__
_silu = ActivationType.Silu
_per_1x32 = QuantType.per_1x32
_fp4x2 = dtypes.fp4x2
_fp8_e8m0 = dtypes.fp8_e8m0
_i32 = dtypes.i32
_fp32 = dtypes.fp32
_empty = torch.empty

@functools.lru_cache(maxsize=2048)
def _get_meta(token, model_dim, inter_dim, expert, topk,
              dtype, hidden_pad, intermediate_pad, is_shuffled):
    md = _orig(token, model_dim, inter_dim, expert, topk,
               dtype, _fp4x2, _fp4x2, _per_1x32, True,
               _silu, False,
               hidden_pad, intermediate_pad, is_shuffled)

    tokens_per_expert = (token * topk) / expert
    use_cktile = False
    sk = 1

    if inter_dim > 1024:
        pass
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
            n_pad_zeros=intermediate_pad // 64 * 64 * 2,
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

_fm.get_2stage_cfgs = lambda *a, **k: _get_meta(a[0], a[1], a[2], a[3], a[4], a[5], a[12], a[13], a[14])

_sorting_bufs = {}
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd


def _do_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_size_M):
    M, topk = topk_ids.shape
    key = (M, E, model_dim, block_size_M)
    if key not in _sorting_bufs:
        device = topk_ids.device
        max_pad = M * topk + E * block_size_M - topk
        max_blk = (max_pad + block_size_M - 1) // block_size_M
        _sorting_bufs[key] = (
            _empty(max_pad, dtype=_i32, device=device),
            _empty(max_pad, dtype=_fp32, device=device),
            _empty(max_blk, dtype=_i32, device=device),
            _empty(2, dtype=_i32, device=device),
            _empty((M, model_dim), dtype=dtype, device=device),
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


def _direct_pipeline(hidden_states, w1, w2, topk,
                     sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                     moe_buf, block_size_M, w1_scale, w2_scale, metadata):
    """Direct per_1x32 + Silu + fp4x2 pipeline without generic branching."""
    token_num = hidden_states.shape[0]
    _, _, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = moe_buf.dtype
    device = hidden_states.device
    is_shuffled = getattr(w1, "is_shuffled", False)
    is_cktile = metadata.ksplit > 1 and is_shuffled

    # Step 1: Quant or skip
    if is_cktile:
        # Cktile: skip quant, pass bf16 directly
        a1 = hidden_states.to(dtype)
        a1_scale = None
    else:
        # CK FP4: quant hidden_states to fp4x2
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=1,
            block_size=block_size_M,
        )

    # Step 2: Allocate intermediate
    a2 = _empty((token_num, topk, inter_dim), dtype=dtype, device=device)

    # Step 3: Stage1 GEMM
    a2 = metadata.stage1(
        a1, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        a2, topk,
        block_m=block_size_M,
        a1_scale=a1_scale,
        w1_scale=w1_scale.view(_fp8_e8m0),
        sorted_weights=None,
    )

    # Step 4: Requant or skip
    if is_cktile:
        a2_scale = None
    else:
        a2 = a2.view(-1, inter_dim)
        a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            topk=topk,
            block_size=block_size_M,
        )
        a2 = a2.view(token_num, topk, -1)

    # Step 5: Stage2 GEMM
    metadata.stage2(
        a2, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_buf, topk,
        w2_scale=w2_scale.view(_fp8_e8m0),
        a2_scale=a2_scale,
        block_m=block_size_M,
        sorted_weights=sorted_weights,
    )


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
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    metadata = _get_meta(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        hidden_states.dtype, hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _do_sorting(
        topk_ids, topk_weights, E, model_dim, hidden_states.dtype, block_size_M,
    )

    _direct_pipeline(
        hidden_states, w1, w2, topk,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_buf, block_size_M, w1_scale, w2_scale, metadata,
    )

    return moe_buf
