"""
Direct 2-stage MoE implementation calling CK stage1/stage2 APIs directly.
Bypasses fused_moe dispatch overhead and uses exact tuned kernel names.
"""
import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import moe_sorting, get_inter_dim
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils

# Tuned kernel names from dsv3_fp4_tuned_fmoe.csv for E=257, d_expert=256
TUNED_KERNELS_E257 = {
    16: {
        "block_m": 32,
        "kn1": "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
        "kn2": "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    },
    128: {
        "block_m": 32,
        "kn1": "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
        "kn2": "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    },
    512: {
        "block_m": 32,
        "kn1": "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
        "kn2": "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    },
}


def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    d_expert = config["d_expert"]
    d_hidden = config["d_hidden"]

    w1 = gate_up_weight_shuffled
    w2 = down_weight_shuffled

    E_dim, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = hidden_states.dtype
    device = hidden_states.device

    # Determine block_m and kernel names
    tuned = TUNED_KERNELS_E257.get(M) if E > 64 else None

    if tuned:
        block_m = tuned["block_m"]
        kn1 = tuned["kn1"]
        kn2 = tuned["kn2"]
    else:
        tokens_per_expert = (M * topk) / E
        if tokens_per_expert < 64:
            block_m = 32
        else:
            block_m = 64
        kn1 = ""
        kn2 = ""

    # Step 1: Token sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )

    # Step 2: Quantize activations and sort scales (fused for small M)
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=M,
        topk=1,
        block_size=block_m,
    )

    # Step 3: Stage 1 GEMM (gate_up)
    w1_scale = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    a2 = torch.empty((M, topk, inter_dim), dtype=dtype, device=device)

    aiter.ck_moe_stage1_fwd(
        a1, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        a2, topk, kn1,
        w1_scale, a1_scale,
        block_m,
        None,  # sorted_weights (doweight_stage1=False)
        QuantType.per_1x32,
        ActivationType.Silu,
        0,     # splitk
        block_m == 32 and (M * topk // E) < 64,  # non_temporal_load (use_nt heuristic)
        dtype,
    )

    # Step 4: Quantize intermediate activations
    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=M,
        topk=topk,
        block_size=block_m,
    )
    a2_q = a2_q.view(M, topk, -1)

    # Step 5: Stage 2 GEMM (down) with weighted reduction
    w2_scale = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)

    aiter.ck_moe_stage2_fwd(
        a2_q, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_buf, topk, kn2,
        w2_scale, a2_scale,
        block_m,
        sorted_weights,  # apply topk_weights in stage2
        QuantType.per_1x32,
        ActivationType.Silu,
        0,     # splitk
        block_m == 32 and (M * topk // E) < 64,  # non_temporal_load
        dtype,
    )

    return moe_buf
