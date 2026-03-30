"""
Direct 2-stage with pre-allocated buffer caching.
Key insight: moe_sorting allocates 5 buffers per call, and we allocate
intermediate buffers. Pre-allocating and reusing eliminates allocation overhead.
Combined with use_non_temporal_load=True and tuned kernel names.
"""
import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import get_inter_dim
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

_TUNED_E257 = {
    16:  ("moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
          "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16", 32),
    64:  ("moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
          "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16", 32),
    128: ("moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
          "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16", 32),
    256: ("moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
          "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16", 32),
    512: ("moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
          "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16", 32),
}

def _pad_m(M):
    if M < 32768:
        p = 1
        while p < M: p *= 2
        return p
    return 32768

# Buffer cache keyed by (M, E, topk, model_dim, inter_dim, block_m)
_buf_cache = {}

def _get_sorting_buffers(M, E, topk, model_dim, block_m, dtype, device):
    key = (M, E, topk, model_dim, block_m)
    if key in _buf_cache:
        return _buf_cache[key]

    max_num_tokens_padded = M * topk + E * block_m - topk
    max_num_m_blocks = (max_num_tokens_padded + block_m - 1) // block_m

    bufs = {
        'sorted_ids': torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device),
        'sorted_weights': torch.empty(max_num_tokens_padded, dtype=torch.float32, device=device),
        'sorted_expert_ids': torch.empty(max_num_m_blocks, dtype=torch.int32, device=device),
        'num_valid_ids': torch.empty(2, dtype=torch.int32, device=device),
        'moe_buf': torch.empty((M, model_dim), dtype=dtype, device=device),
    }
    _buf_cache[key] = bufs
    return bufs

# Intermediate buffer cache
_inter_cache = {}

def _get_inter_buffer(M, topk, inter_dim, dtype, device):
    key = (M, topk, inter_dim)
    if key not in _inter_cache:
        _inter_cache[key] = torch.empty((M, topk, inter_dim), dtype=dtype, device=device)
    return _inter_cache[key]


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype, device = hidden_states.dtype, hidden_states.device

    padded_M = _pad_m(M)
    tuned = _TUNED_E257.get(padded_M) if E > 64 else None
    if tuned:
        kn1, kn2, block_m = tuned
    else:
        tokens_per_expert = (M * topk) / E
        block_m = 32 if tokens_per_expert < 64 else 64
        kn1, kn2 = "", ""

    use_nt = True  # Force NT for sparse MoE patterns

    # Pre-allocated sorting buffers
    bufs = _get_sorting_buffers(M, E, topk, model_dim, block_m, dtype, device)
    sorted_ids = bufs['sorted_ids']
    sorted_weights = bufs['sorted_weights']
    sorted_expert_ids = bufs['sorted_expert_ids']
    num_valid_ids = bufs['num_valid_ids']
    moe_buf = bufs['moe_buf']

    # Sorting (writes into pre-allocated buffers)
    aiter.moe_sorting_fwd(
        topk_ids, topk_weights,
        sorted_ids, sorted_weights, sorted_expert_ids,
        num_valid_ids, moe_buf,
        E, block_m,
    )

    # Fused quantize + sort
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=block_m,
    )

    # Stage 1 GEMM with pre-allocated intermediate buffer
    w1_scale = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    a2 = _get_inter_buffer(M, topk, inter_dim, dtype, device)

    aiter.ck_moe_stage1_fwd(
        a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
        a2, topk, kn1, w1_scale, a1_scale, block_m,
        None, QuantType.per_1x32, ActivationType.Silu, 0, use_nt, dtype,
    )

    # Fused quantize intermediate + sort
    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=topk, block_size=block_m,
    )
    a2_q = a2_q.view(M, topk, -1)

    # Stage 2 GEMM
    w2_scale = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    aiter.ck_moe_stage2_fwd(
        a2_q, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_buf, topk,
        kernelName=kn2, w2_scale=w2_scale, a2_scale=a2_scale,
        block_m=block_m, sorted_weights=sorted_weights,
        quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
        use_non_temporal_load=use_nt,
    )

    return moe_buf
