"""
Expert-parallel approach: process each expert's tokens independently using
the Triton GEMM kernel (gemm_afp4wfp4_preshuffle) instead of CK MoE kernels.

For E=33 shapes, each expert processes ~4-140 tokens. The Triton GEMM is
well-tuned for small M from the mxfp4-mm leaderboard work.

This approach:
1. Groups tokens by expert
2. For each active expert: GEMM1(gate_up) -> SwiGLU -> GEMM2(down)
3. Weighted accumulation into output

Avoids CK MoE overhead but has more Python-level dispatch + lacks fusion.
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from aiter.utility.fp4_utils import e8m0_shuffle


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
    out = (silu_gate * up).to(tl.bfloat16)
    tl.store(out_ptr + offs, out, mask=mask)


def _swiglu(gate_up, d_expert):
    """Apply SwiGLU: SiLU(gate) * up. gate_up is [M, 2*d_expert]."""
    gate = gate_up[:, :d_expert].contiguous()
    up = gate_up[:, d_expert:].contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _swiglu_kernel[grid](gate, up, out, n, BLOCK)
    return out


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

    # For large E (E=257), use the proven fused_moe path
    if E > 64:
        hidden_pad = config["d_hidden_pad"] - d_hidden
        intermediate_pad = config["d_expert_pad"] - d_expert
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

    # Expert-parallel for E=33 shapes
    device = hidden_states.device

    # Count tokens per expert
    flat_ids = topk_ids.view(-1)  # [M * topk]
    output = torch.zeros((M, d_hidden), dtype=torch.bfloat16, device=device)

    # Quantize all hidden states once
    h_q, h_scale = dynamic_mxfp4_quant(hidden_states)
    # Shuffle scale for preshuffle API (M >= 32)
    if M >= 32:
        h_scale_sh = e8m0_shuffle(h_scale)
    else:
        h_scale_sh = h_scale

    # Process each expert
    for expert_id in range(E):
        # Find which (token, slot) pairs route to this expert
        mask_flat = (flat_ids == expert_id)
        if not mask_flat.any():
            continue

        indices = mask_flat.nonzero(as_tuple=True)[0]  # indices into flat_ids
        token_indices = indices // topk
        slot_indices = indices % topk
        n_tokens = len(token_indices)

        # Gather quantized activations for this expert's tokens
        x_q = h_q[token_indices]  # [n_tokens, K//2]
        x_scale = h_scale_sh[token_indices] if M >= 32 else h_scale[token_indices]

        # Get expert weights (already shuffled)
        w1_e = gate_up_weight_shuffled[expert_id]  # [2*d_expert_pad, d_hidden_pad//2] shuffled
        w1_s = gate_up_weight_scale_shuffled  # flat shuffled scales - need per-expert slice
        w2_e = down_weight_shuffled[expert_id]
        w2_s = down_weight_scale_shuffled

        # This approach won't easily work because the shuffled weight scales
        # are in a flat format that's hard to slice per-expert.
        # Fall back to fused_moe for correctness.
        break

    # Fall back to fused_moe with optimized block_m
    hidden_pad = config["d_hidden_pad"] - d_hidden
    intermediate_pad = config["d_expert_pad"] - d_expert
    tokens_per_expert = (M * topk) / E
    block_m = 32 if tokens_per_expert < 64 else 64

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
        block_size_M=block_m,
    )
