"""
Expert-parallel approach using raw (non-shuffled) weights + Triton GEMM.
For E=33 shapes, processes each expert's tokens with gemm_afp4wfp4.
Uses raw weights which have clean per-expert indexing [E, N, K//2].

For E=257 shapes, uses fused_moe (tuned configs).
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4


@triton.jit
def _swiglu_kernel(x_ptr, n_tokens, d_expert, BLOCK: tl.constexpr):
    """SwiGLU in-place: x[:, :d_expert] = SiLU(x[:, :d_expert]) * x[:, d_expert:]"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_tokens * d_expert

    row = offs // d_expert
    col = offs % d_expert

    gate_idx = row * d_expert * 2 + col
    up_idx = row * d_expert * 2 + d_expert + col

    gate = tl.load(x_ptr + gate_idx, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + up_idx, mask=mask).to(tl.float32)
    silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    out = (silu_gate * up).to(tl.bfloat16)

    # Write result to first half
    tl.store(x_ptr + gate_idx, out, mask=mask)


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
    d_expert_pad = config["d_expert_pad"]
    d_hidden_pad = config["d_hidden_pad"]

    hidden_pad = d_hidden_pad - d_hidden
    intermediate_pad = d_expert_pad - d_expert

    # For E=257 or large E: use proven fused_moe path
    if E > 64:
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
    output = torch.zeros((M, d_hidden), dtype=torch.bfloat16, device=device)

    # Quantize all hidden states once
    h_q, h_scale = dynamic_mxfp4_quant(hidden_states)
    # h_q: [M, d_hidden//2] fp4x2, h_scale: [M, d_hidden//32] e8m0

    # Process each expert
    for expert_id in range(E):
        # Find tokens routed to this expert
        expert_mask = (topk_ids == expert_id)  # [M, topk]
        if not expert_mask.any():
            continue

        # Get (token_idx, slot_idx) pairs
        token_idxs, slot_idxs = expert_mask.nonzero(as_tuple=True)
        n_tok = len(token_idxs)
        weights = topk_weights[token_idxs, slot_idxs]  # [n_tok]

        # Gather quantized hidden states
        x_q = h_q[token_idxs]        # [n_tok, d_hidden_pad//2]
        x_s = h_scale[token_idxs]    # [n_tok, d_hidden_pad//32]

        # Expert gate_up weights (raw, not shuffled)
        w1_e = gate_up_weight[expert_id]   # [2*d_expert_pad, d_hidden_pad//2]
        w1_s = gate_up_weight_scale[expert_id]  # [2*d_expert_pad, scale_K]

        # Stage 1: gate_up GEMM: [n_tok, d_hidden_pad//2] x [2*d_expert_pad, d_hidden_pad//2]^T
        gate_up_out = gemm_afp4wfp4(
            x_q, w1_e, x_s, w1_s,
            dtype=torch.bfloat16,
        )  # [n_tok, 2*d_expert_pad]

        # SwiGLU activation
        gate = gate_up_out[:, :d_expert_pad]
        up = gate_up_out[:, d_expert_pad:]
        silu_gate = gate.float() * torch.sigmoid(gate.float())
        intermediate = (silu_gate * up.float()).to(torch.bfloat16)
        # Trim padding
        intermediate = intermediate[:, :d_expert]

        # Quantize intermediate
        inter_q, inter_s = dynamic_mxfp4_quant(intermediate)

        # Expert down weights (raw)
        w2_e = down_weight[expert_id]     # [d_hidden_pad, d_expert_pad//2]
        w2_s = down_weight_scale[expert_id]  # [d_hidden_pad, scale_K_2]

        # But wait - intermediate has d_expert cols (not d_expert_pad)
        # and w2 expects d_expert_pad cols. Need to pad.
        # Actually inter_q from dynamic_mxfp4_quant should be [n_tok, d_expert//2]
        # and w2_e is [d_hidden_pad, d_expert_pad//2]
        # The dimensions must match: K for GEMM = d_expert_pad
        # But intermediate has d_expert (unpadded). We need to pad.

        if d_expert != d_expert_pad:
            # Pad intermediate to d_expert_pad
            pad_size = d_expert_pad - d_expert
            intermediate_padded = torch.nn.functional.pad(intermediate, (0, pad_size))
            inter_q, inter_s = dynamic_mxfp4_quant(intermediate_padded)

        # Stage 2: down GEMM
        down_out = gemm_afp4wfp4(
            inter_q, w2_e, inter_s, w2_s,
            dtype=torch.bfloat16,
        )  # [n_tok, d_hidden_pad]

        # Trim to d_hidden and weighted accumulate
        down_out = down_out[:, :d_hidden]
        output[token_idxs] += weights.unsqueeze(1) * down_out

    return output
