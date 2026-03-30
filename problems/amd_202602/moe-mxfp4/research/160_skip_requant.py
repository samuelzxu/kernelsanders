"""
v160: CK FP4 stage 1 + cktile BF16 stage 2, skipping intermediate requantization.
The a16w4 approach: stage 1 outputs BF16, stage 2 consumes BF16 directly.
Only applies to dense shapes where CK FP4 stage 1 is used.
Sparse shapes still use full cktile path (already no requant).
"""
import os, sys, stat
_JIT_DIR = "/home/runner/aiter/aiter/jit"
_BASE_URL = "https://github.com/samuelzxu/aiter-precompiled/releases/download/v0.3-rocm71"
_MODULES = ["module_aiter_enum.so","module_moe_sorting_opus.so","module_moe_sorting.so",
    "module_quant.so","module_activation.so","module_moe_cktile2stages.so",
    "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so"]
def _install():
    import urllib.request
    os.makedirs(_JIT_DIR, exist_ok=True)
    for name in _MODULES:
        path = os.path.join(_JIT_DIR, name)
        if not os.path.exists(path):
            try: urllib.request.urlretrieve(f"{_BASE_URL}/{name}", path); os.chmod(path, 0o755)
            except Exception: pass
try: _install()
except Exception: pass

import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_USE_NT'] = '1'

import torch, functools, aiter
from task import input_t, output_t
from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe_2stages, get_inter_dim, get_padded_M, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2,
    get_quant, fused_dynamic_mxfp4_quant_moe_sort,
)
from aiter.utility import fp4_utils

_orig = get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk,
             dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
             activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled):
    md = _orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w,
               q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled)
    tokens_per_expert = (token * topk) / expert
    use_cktile = False; sk = 1
    if inter_dim > 1024: use_cktile = False
    elif tokens_per_expert < 5: use_cktile = True; sk = 2
    elif tokens_per_expert < 40 and expert <= 33: use_cktile = True; sk = 1
    if use_cktile and is_shuffled:
        md.ksplit = 2
        md.block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        md.stage1 = functools.partial(cktile_moe_stage1,
            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
            k_pad_zeros=hidden_pad // 128 * 128, activation=ActivationType.Silu, split_k=sk)
        md.stage2 = functools.partial(cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64, k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu)
    return md
_fm.get_2stage_cfgs = _patched

_sorting_bufs = {}
_has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
_sort_fwd = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd
def _fast_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_size_M):
    M, topk = topk_ids.shape
    key = (M, E, model_dim, block_size_M)
    if key not in _sorting_bufs:
        device = topk_ids.device
        max_num_tokens_padded = M * topk + E * block_size_M - topk
        max_num_m_blocks = (max_num_tokens_padded + block_size_M - 1) // block_size_M
        _sorting_bufs[key] = (
            torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
            torch.empty(2, dtype=dtypes.i32, device=device),
            torch.empty((M, model_dim), dtype=dtype, device=device))
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[key]
    _sort_fwd(topk_ids, topk_weights, sorted_ids, sorted_weights,
              sorted_expert_ids, num_valid_ids, moe_buf, E, int(block_size_M), None, None, 0)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states, _, _, _, _, gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    M, topk = topk_ids.shape
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    isG1U1 = inter_dim != w1.shape[1]
    dtype = hidden_states.dtype
    is_shuffled = getattr(w1, "is_shuffled", False)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    metadata = _patched(get_padded_M(M), model_dim, inter_dim, E, topk,
                        dtype, dtypes.fp4x2, dtypes.fp4x2, QuantType.per_1x32, isG1U1,
                        ActivationType.Silu, False, hidden_pad, intermediate_pad, is_shuffled)
    block_size_M = int(metadata.block_m)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fast_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_size_M)

    # Check if this shape uses cktile (already skips requant) or CK FP4
    tokens_per_expert = (M * topk) / E
    use_cktile = (inter_dim <= 1024 and
                  (tokens_per_expert < 5 or (tokens_per_expert < 40 and E <= 33)))

    if use_cktile:
        # Standard cktile path (already optimal, no requant)
        fused_moe_2stages(hidden_states, w1, w2, topk, sorted_ids, sorted_weights,
                          sorted_expert_ids, num_valid_ids, moe_buf, isG1U1, block_size_M,
                          activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                          doweight_stage1=False, q_dtype_a=dtypes.fp4x2, q_dtype_w=dtypes.fp4x2,
                          w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled,
                          a1_scale=None, a2_scale=None,
                          hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
    else:
        # Dense shape: CK FP4 stage 1 + cktile BF16 stage 2 (SKIP REQUANT)
        # Step 1: Quant activations
        quant_func = get_quant(QuantType.per_1x32)
        if M <= 1024:
            a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                token_num=M, topk=1, block_size=block_size_M)
        else:
            a1, a1_scale = quant_func(hidden_states, scale=None, quant_dtype=dtypes.fp4x2)
            a1_scale = fp4_utils.moe_mxfp4_sort(a1_scale, sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids, token_num=M, block_size=block_size_M)

        # Step 2: CK FP4 stage 1 (outputs BF16 intermediate a2)
        a2 = torch.empty((M, topk, inter_dim), dtype=dtype, device=hidden_states.device)
        metadata.stage1(
            a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
            a1_scale=a1_scale,
            w1_scale=gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0),
            block_m=block_size_M)

        # Step 3: cktile BF16 stage 2 (NO REQUANT — a2 is BF16!)
        cktile_stage2 = functools.partial(cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64,
            k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu)
        cktile_stage2(
            a2, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_buf, topk,
            w2_scale=down_weight_scale_shuffled.view(dtypes.fp8_e8m0),
            a2_scale=None,
            block_m=block_size_M,
            sorted_weights=sorted_weights)

    return moe_buf
