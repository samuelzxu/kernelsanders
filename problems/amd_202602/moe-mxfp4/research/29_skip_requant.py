"""
Skip intermediate FP4 requantization by using cktile_moe_gemm2 for stage2.
Pipeline: sort → quant_input → CK_stage1 → cktile_stage2(bf16_intermediate)
This eliminates one Triton kernel launch + global memory round-trip.

The cktile_moe_gemm2 can accept bf16 intermediates directly.
"""
import torch
import functools
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import moe_sorting, get_inter_dim
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

# Monkey-patch use_nt
import aiter.fused_moe as _fm
_fm.use_nt = functools.lru_cache(maxsize=2048)(lambda token, topk, e: True)

_TUNED_E257 = {
    16:  ("moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16", 32),
    64:  ("moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16", 32),
    128: ("moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16", 32),
    256: ("moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16", 32),
    512: ("moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16", 32),
}

def _pad_m(M):
    if M < 32768:
        p = 1
        while p < M: p *= 2
        return p
    return 32768


@torch.inference_mode()
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
    d_expert = config["d_expert"]
    d_hidden = config["d_hidden"]
    hidden_pad = config["d_hidden_pad"] - d_hidden
    intermediate_pad = config["d_expert_pad"] - d_expert

    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype, device = hidden_states.dtype, hidden_states.device

    padded_M = _pad_m(M)
    tuned = _TUNED_E257.get(padded_M) if E > 64 else None
    if tuned:
        kn1, block_m = tuned
    else:
        tokens_per_expert = (M * topk) / E
        block_m = 32 if tokens_per_expert < 64 else 64
        kn1 = ""

    use_nt = True

    # Step 1: Sort
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )

    # Step 2: Quantize input + sort scales (fused Triton kernel)
    w1_scale = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=block_m,
    )

    # Step 3: Stage1 GEMM (outputs bf16 intermediate with SwiGLU)
    a2 = torch.empty((M, topk, inter_dim), dtype=dtype, device=device)
    aiter.ck_moe_stage1_fwd(
        a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
        a2, topk, kn1, w1_scale, a1_scale, block_m,
        None, QuantType.per_1x32, ActivationType.Silu, 0, use_nt, dtype,
    )

    # Step 4: Stage2 using cktile with bf16 intermediate (SKIP requant!)
    # cktile_moe_gemm2 accepts bf16 directly
    w2_scale = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    try:
        aiter.moe_cktile2stages_gemm2(
            a2, w2, moe_buf,
            sorted_ids, sorted_expert_ids, num_valid_ids,
            topk,
            hidden_pad // 64 * 64,     # n_padded_zeros
            intermediate_pad // 128 * 128,  # k_padded_zeros
            sorted_weights,  # topk_weight for weighted reduction
            None,           # x_scale (bf16, no scale)
            w2_scale,       # w_scale
            None,           # exp_bias
            ActivationType.Silu,
            block_m,
        )
    except Exception:
        # Fallback: standard 2-stage with requantization
        a2_flat = a2.view(-1, inter_dim)
        a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=topk, block_size=block_m,
        )
        a2_q = a2_q.view(M, topk, -1)
        aiter.ck_moe_stage2_fwd(
            a2_q, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
            moe_buf, topk,
            kernelName="", w2_scale=w2_scale, a2_scale=a2_scale,
            block_m=block_m, sorted_weights=sorted_weights,
            quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
            use_non_temporal_load=use_nt,
        )

    return moe_buf
