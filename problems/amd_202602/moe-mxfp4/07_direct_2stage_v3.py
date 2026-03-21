"""
Direct 2-stage MoE with corrected API signatures.
stage1: 18 positional args (including splitk, non_temporal_load, dst_type)
stage2: keyword args only after topk (no splitk, no dst_type)
Uses pre-allocated buffers to reduce allocation overhead.
"""
import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import moe_sorting, get_inter_dim
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort


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

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]

    w1 = gate_up_weight_shuffled
    w2 = down_weight_shuffled

    _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype = hidden_states.dtype
    device = hidden_states.device

    # Shape-aware block_m
    tokens_per_expert = (M * topk) / E
    if tokens_per_expert < 64:
        block_m = 32
    else:
        block_m = 64

    use_nt = (M * topk // E) < 64

    # Step 1: Token sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )

    # Step 2: Fused quantize + sort scales
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
        a2, topk, "",
        w1_scale, a1_scale,
        block_m,
        None,  # sorted_weights (doweight_stage1=False)
        QuantType.per_1x32,
        ActivationType.Silu,
        0,      # splitk
        use_nt, # non_temporal_load
        dtype,  # dst_type
    )

    # Step 4: Fused quantize intermediate + sort
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
    # stage2 signature: 8 positional + kwargs (no splitk, no dst_type)
    w2_scale = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)

    aiter.ck_moe_stage2_fwd(
        a2_q, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_buf, topk,
        kernelName="",
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        block_m=block_m,
        sorted_weights=sorted_weights,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
        use_non_temporal_load=use_nt,
    )

    return moe_buf
