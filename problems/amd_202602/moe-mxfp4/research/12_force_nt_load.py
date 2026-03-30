"""
Direct 2-stage with forced use_non_temporal_load=True for all shapes.
The tuned configs default to use_nt=False, but our shapes are sparse
(tokens_per_expert < 64 for E=257), so non-temporal loads should help.
Combined with tuned kernel names for E=257 and block_m tuning for E=33.
"""
import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import moe_sorting, get_inter_dim
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

    # Force non-temporal load for sparse MoE patterns
    use_nt = True

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )

    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=block_m,
    )

    w1_scale = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    a2 = torch.empty((M, topk, inter_dim), dtype=dtype, device=device)

    aiter.ck_moe_stage1_fwd(
        a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
        a2, topk, kn1, w1_scale, a1_scale, block_m,
        None, QuantType.per_1x32, ActivationType.Silu, 0, use_nt, dtype,
    )

    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=topk, block_size=block_m,
    )
    a2_q = a2_q.view(M, topk, -1)

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
