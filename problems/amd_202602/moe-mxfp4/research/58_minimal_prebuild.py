"""
v58: Minimal prebuild - only trigger module_moe_asm build.
Avoid triggering module_quant build (uses Triton quant instead).
The fused_moe_1stage path uses Triton's fused_dynamic_mxfp4_quant_moe_sort
internally, not HIP quant. So module_quant is not needed.

Key: call fmoe_g1u1 directly with pre-quantized data to avoid
triggering any unnecessary module builds.
"""
import torch
import functools
import sys
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter
import aiter.fused_moe as _fm
from aiter.fused_moe import moe_sorting
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility import fp4_utils

_build_done = False

# Monkey-patch: replace HIP quant with Triton quant in fused_moe_1stage
# This avoids triggering module_quant build (saves ~25s)
def _triton_quant_wrapper(x, scale=None, quant_dtype=None, shuffle=False, num_rows=None, num_rows_factor=1):
    """Triton-based quant that matches HIP quant API."""
    fp4, bs = dynamic_mxfp4_quant(x)
    return fp4.view(dtypes.fp4x2), bs.view(dtypes.fp8_e8m0)

# Replace get_quant in fused_moe module to return Triton quant for per_1x32
_original_get_quant = _fm.get_quant
def _patched_get_quant(quant_type):
    if quant_type == QuantType.per_1x32:
        return _triton_quant_wrapper
    return _original_get_quant(quant_type)
_fm.get_quant = _patched_get_quant

def _trigger_asm_build():
    """Trigger module_moe_asm build using Triton quant (not HIP quant)."""
    global _build_done
    if _build_done:
        return
    _build_done = True

    try:
        M, E, topk, d = 2, 3, 2, 128
        device = "cuda"
        hidden = torch.randn(M, d, dtype=torch.bfloat16, device=device)
        topk_ids = torch.zeros(M, topk, dtype=torch.int32, device=device)
        topk_w = torch.ones(M, topk, dtype=torch.float32, device=device) / topk

        # Sort (triggers module_moe_sorting build - already needed by 2-stage)
        sids, sw, seids, nvids, mbuf = moe_sorting(topk_ids, topk_w, E, d, torch.bfloat16, 32)

        # Quant using TRITON (no module_quant build needed)
        a1, a1s = dynamic_mxfp4_quant(hidden)
        a1 = a1.view(dtypes.fp4x2)
        a1s = a1s.view(dtypes.fp8_e8m0)
        a1s = fp4_utils.moe_mxfp4_sort(a1s, sids, nvids, M, 32)

        # Create minimal fp4x2 weights
        w1 = torch.zeros(E, d, d // 2, dtype=torch.uint8, device=device).view(dtypes.fp4x2)
        w2 = torch.zeros(E, d, d // 2, dtype=torch.uint8, device=device).view(dtypes.fp4x2)
        ws = torch.ones(E, d * d // 32, dtype=torch.uint8, device=device).view(dtypes.fp8_e8m0)

        # Call fmoe_g1u1 to trigger module_moe_asm build ONLY
        aiter.fmoe_g1u1(mbuf, a1, w1, w2, sids, sw, seids, nvids,
                         topk, a1s, ws, ws, "",
                         fc2_smooth_scale=None, activation=ActivationType.Silu)
        print("[PREBUILD] module_moe_asm built OK", file=sys.stderr)
    except Exception as e:
        print(f"[PREBUILD] Build triggered ({type(e).__name__})", file=sys.stderr)
    torch.cuda.empty_cache()


# Force 1-stage for inter_dim <= 1024
_orig = _fm.get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk, *rest, **kw):
    md = _orig(token, model_dim, inter_dim, expert, topk, *rest, **kw)
    if inter_dim <= 1024:
        md.run_1stage = True
        md.stage1 = functools.partial(
            _fm.fused_moe_1stage,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
        )
    return md

_fm.get_2stage_cfgs = _patched


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    # Trigger ASM build on first call
    if not _build_done:
        _trigger_asm_build()

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return _fm.fused_moe(
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
