"""
Fused bf16→fp4 quantization + GEMM for stage1.
Instead of separate quant kernel + stage1 CK kernel, this attempts to:
1. Keep the CK stage1 for the GEMM (can't beat assembly)
2. But quantize directly from bf16 using dynamic_mxfp4_quant (no sorting!)
   and pass pre-sorted scales separately via moe_mxfp4_sort

The key insight from profiling: fused_dynamic_mxfp4_quant_moe_sort does TWO
things (quant + sort) in one kernel. But we can split them IF the separate
path is faster. The quant kernel alone might be faster than the fused version
because it doesn't need to handle the sorted_ids indexing.
"""
import torch
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_inter_dim
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]

    # For E=257 with tuned configs, use fused_moe directly (it's optimal)
    if E > 64:
        return fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    # For E=33: try separate quant + sort path
    w1, w2 = gate_up_weight_shuffled, down_weight_shuffled
    _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    dtype, device = hidden_states.dtype, hidden_states.device

    tokens_per_expert = (M * topk) / E
    block_m = 32 if tokens_per_expert < 64 else 64

    # Step 1: Sort
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m,
    )

    # Step 2: Separate quant (Triton) + sort (Triton) instead of fused
    # dynamic_mxfp4_quant: just quantize, no sorting
    a1, a1_scale_raw = dynamic_mxfp4_quant(hidden_states)
    # View as fp4x2 dtype (CK expects this, not uint8)
    a1 = a1.view(dtypes.fp4x2)
    a1_scale_raw = a1_scale_raw.view(dtypes.fp8_e8m0)
    # moe_mxfp4_sort: just sort the scales
    a1_scale = fp4_utils.moe_mxfp4_sort(
        a1_scale_raw, sorted_ids, num_valid_ids, M, block_m,
    )

    # Step 3: Stage 1 GEMM
    w1_scale = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    a2 = torch.empty((M, topk, inter_dim), dtype=dtype, device=device)

    aiter.ck_moe_stage1_fwd(
        a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
        a2, topk, "", w1_scale, a1_scale, block_m,
        None, QuantType.per_1x32, ActivationType.Silu, 0, True, dtype,
    )

    # Step 4: Fused quant+sort for intermediate (keep fused - it's optimized)
    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=topk, block_size=block_m,
    )
    a2_q = a2_q.view(M, topk, -1)

    # Step 5: Stage 2 GEMM
    w2_scale = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    aiter.ck_moe_stage2_fwd(
        a2_q, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_buf, topk,
        kernelName="", w2_scale=w2_scale, a2_scale=a2_scale,
        block_m=block_m, sorted_weights=sorted_weights,
        quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
        use_non_temporal_load=True,
    )

    return moe_buf
