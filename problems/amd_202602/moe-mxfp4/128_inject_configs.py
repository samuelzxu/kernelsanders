"""
v128: Inject tuned kernel configs for shapes missing from CSV.
The CSV only has E=257 entries; E=33 shapes use heuristic fallback.
Also E=257 bs=512 uses wrong 64x32 kernel (should be 256x32).
Inject optimal configs by monkey-patching cfg_2stages dict directly.
Uses 256x32x128x128_1x4 (large tile) for dense shapes where it helps.
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

# Kernel name constants
_K64 = 'moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16'
_K256 = 'moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16'
_K2_64 = 'moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16'

# Inject configs into the global cfg_2stages dict
# Force trigger config loading first
_dummy_md = get_2stage_cfgs(
    256, 7168, 256, 257, 9,
    dtypes.bf16, dtypes.fp4x2, dtypes.fp4x2,
    QuantType.per_1x32, True,
    ActivationType.Silu, False, 0, 0, True,
)

# Now patch the global config dict with better kernel selections
if _fm.cfg_2stages is not None:
    # E=257 bs=512 (token=512): force 256x kernel for stage1
    _fm.cfg_2stages[(256, 512, 7168, 256, 257, 9,
        'ActivationType.Silu', 'torch.bfloat16',
        'torch.float4_e2m1fn_x2', 'torch.float4_e2m1fn_x2',
        'QuantType.per_1x32', True, False)] = {
        'block_m': 64, 'ksplit': 0,
        'kernelName1': _K256, 'kernelName2': _K2_64,
        'run_1stage': False,
    }

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
            torch.empty((M, model_dim), dtype=dtype, device=device),
        )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sorting_bufs[key]
    _sort_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids,
        moe_buf, E, int(block_size_M),
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

    metadata = _patched(
        get_padded_M(M), model_dim, inter_dim, E, topk,
        dtype, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, isG1U1,
        ActivationType.Silu, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_size_M = int(metadata.block_m)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _fast_sorting(
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
        q_dtype_a=dtypes.fp4x2, q_dtype_w=dtypes.fp4x2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    return moe_buf
