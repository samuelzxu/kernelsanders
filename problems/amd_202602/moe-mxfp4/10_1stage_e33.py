"""
Hybrid: Use 1-stage fmoe_g1u1 for E=33 shapes (avoids intermediate quantization),
and default fused_moe for E=257 shapes (tuned configs).

The 1-stage path does: sort → GEMM1 → activation → GEMM2 → weighted reduction
all in one kernel, avoiding intermediate buffer allocation and requantization.
"""
import torch
import functools
from task import input_t, output_t

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_inter_dim, fused_moe_1stage
from aiter.utility import fp4_utils
from aiter import get_hip_quant as get_quant


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

    w1 = gate_up_weight_shuffled
    w2 = down_weight_shuffled

    if E > 64:
        # E=257: use tuned 2-stage configs
        output = fused_moe(
            hidden_states, w1, w2,
            topk_weights, topk_ids,
            expert_mask=None,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
        )
    else:
        # E=33: try 1-stage path
        _, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
        dtype = hidden_states.dtype
        device = hidden_states.device

        tokens_per_expert = (M * topk) / E
        block_m = 32 if tokens_per_expert < 64 else 64

        # Sorting
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
            topk_ids, topk_weights, E, model_dim, dtype, block_m,
        )

        # Quantize to fp4
        quant_func = get_quant(QuantType.per_1x32)
        a1, a1_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=dtypes.fp4x2,
        )

        # Sort the scales
        a1_scale = fp4_utils.moe_mxfp4_sort(
            a1_scale,
            sorted_ids,
            num_valid_ids,
            M,
            block_m,
        )

        w1_scale = gate_up_weight_scale_shuffled.view(E, -1)
        w2_scale = down_weight_scale_shuffled.view(E, -1)

        # Call 1-stage fmoe_g1u1
        fused_moe_1stage(
            a1, w1, w2,
            topk, sorted_ids, sorted_weights,
            sorted_expert_ids, num_valid_ids,
            moe_buf, True,  # isG1U1
            block_m,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            q_dtype_a=dtypes.fp4x2,
            q_dtype_w=dtypes.fp4x2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
        )
        output = moe_buf

    return output
