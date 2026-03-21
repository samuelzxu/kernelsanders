"""
Expert-parallel MoE using raw weights + correct scale slicing.
Raw scales are [E*N, K//32] 2D - slice as scale[expert_id*N:(expert_id+1)*N, :].
Uses gemm_afp4wfp4 (Triton GEMM) per expert - no CK assembly JIT needed.

For E=257 shapes: use fused_moe (tuned CK kernels, pre-compiled)
For E=33 shapes: try expert-parallel (fewer experts, larger per-expert GEMM)
"""
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


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
    d_expert = config["d_expert"]
    d_expert_pad = config["d_expert_pad"]
    d_hidden = config["d_hidden"]
    d_hidden_pad = config["d_hidden_pad"]

    M = hidden_states.shape[0]
    E = gate_up_weight.shape[0]
    topk = topk_ids.shape[1]

    # For large E or large d_expert, use proven fused_moe with CK kernels
    if E > 64 or d_expert > 1024:
        return fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids,
            expert_mask=None,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    # Expert-parallel for E=33 shapes using raw weights
    device = hidden_states.device
    N1 = gate_up_weight.shape[1]  # 2*d_expert_pad
    N2 = down_weight.shape[1]     # d_hidden_pad
    K1 = gate_up_weight.shape[2]  # d_hidden_pad//2 (fp4x2 packed)
    K2 = down_weight.shape[2]     # d_expert_pad//2 (fp4x2 packed)

    output = torch.zeros((M, d_hidden), dtype=torch.bfloat16, device=device)

    # Quantize all hidden states once
    h_q, h_scale = dynamic_mxfp4_quant(hidden_states)

    # Process each expert
    for expert_id in range(E):
        # Find tokens routed to this expert
        mask = (topk_ids == expert_id)  # [M, topk]
        if not mask.any():
            continue

        token_idxs, slot_idxs = mask.nonzero(as_tuple=True)
        n_tok = len(token_idxs)
        weights = topk_weights[token_idxs, slot_idxs]  # [n_tok]

        # Gather quantized activations
        x_q = h_q[token_idxs]      # [n_tok, K1] fp4x2
        x_s = h_scale[token_idxs]  # [n_tok, K1*2//32] e8m0

        # Get expert's gate_up weights and scales (raw format)
        # View as uint8 - gemm_afp4wfp4 expects uint8 packed format
        w1_e = gate_up_weight[expert_id].view(torch.uint8)  # [N1, K1] uint8
        # Scales are [E*N1, K1*2//32] - slice for this expert
        w1_s = gate_up_weight_scale[expert_id * N1 : (expert_id + 1) * N1, :].view(torch.uint8)  # [N1, K1*2//32]

        # Stage 1: gate_up GEMM: [n_tok, K1] x [N1, K1]^T -> [n_tok, N1]
        # Use explicit config to avoid bad autotuning for small M
        gemm_config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 1,
        }
        gate_up_out = gemm_afp4wfp4(
            x_q.view(torch.uint8), w1_e, x_s.view(torch.uint8), w1_s,
            dtype=torch.bfloat16, config=gemm_config,
        )

        # SwiGLU activation: silu(gate) * up
        gate = gate_up_out[:, :d_expert_pad].float()
        up = gate_up_out[:, d_expert_pad:].float()
        intermediate = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

        # Quantize intermediate for stage 2
        inter_q, inter_s = dynamic_mxfp4_quant(intermediate)

        # Get expert's down weights and scales
        w2_e = down_weight[expert_id].view(torch.uint8)  # [N2, K2] uint8
        w2_s = down_weight_scale[expert_id * N2 : (expert_id + 1) * N2, :].view(torch.uint8)  # [N2, K2*2//32]

        # Stage 2: down GEMM: [n_tok, K2] x [N2, K2]^T -> [n_tok, N2]
        down_out = gemm_afp4wfp4(
            inter_q.view(torch.uint8), w2_e, inter_s.view(torch.uint8), w2_s,
            dtype=torch.bfloat16, config=gemm_config,
        )

        # Weighted accumulation (trim padding)
        output[token_idxs] += weights.unsqueeze(1) * down_out[:, :d_hidden]

    return output
