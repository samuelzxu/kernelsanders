"""
Pre-build assembly module inline (same process) at import time.
The eval.py spawns a multiprocessing.Pool worker that imports this module.
All import-time code runs before the first custom_kernel call.
"""
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter
import aiter.fused_moe as _fm
from aiter.fused_moe import moe_sorting, BLOCK_SIZE_M
from aiter import get_hip_quant
from aiter.utility import fp4_utils
import functools
import sys

# Pre-build: trigger module_moe_asm compilation at import time
def _prebuild():
    try:
        M, E, topk, d = 2, 3, 2, 128
        hidden = torch.randn(M, d, dtype=torch.bfloat16, device="cuda")
        topk_ids = torch.zeros(M, topk, dtype=torch.int32, device="cuda")
        topk_w = torch.ones(M, topk, dtype=torch.float32, device="cuda") / topk

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
            topk_ids, topk_w, E, d, torch.bfloat16, 32,
        )

        quant_func = get_hip_quant(QuantType.per_1x32)
        a1, a1_scale = quant_func(hidden, scale=None, quant_dtype=dtypes.fp4x2)
        a1_scale = fp4_utils.moe_mxfp4_sort(a1_scale, sorted_ids, num_valid_ids, M, 32)

        w1 = torch.zeros(E, d, d // 2, dtype=torch.uint8, device="cuda").view(dtypes.fp4x2)
        w2 = torch.zeros(E, d, d // 2, dtype=torch.uint8, device="cuda").view(dtypes.fp4x2)
        ws1 = torch.zeros(E, d * d // 32, dtype=torch.uint8, device="cuda").view(dtypes.fp8_e8m0)
        ws2 = torch.zeros(E, d * d // 32, dtype=torch.uint8, device="cuda").view(dtypes.fp8_e8m0)

        aiter.fmoe_g1u1(
            moe_buf, a1, w1, w2,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            topk, a1_scale, ws1, ws2,
            "", fc2_smooth_scale=None, activation=ActivationType.Silu,
        )
        print("[PREBUILD] module_moe_asm built successfully", file=sys.stderr)
    except Exception as e:
        print(f"[PREBUILD] module_moe_asm build triggered (may have errored on dummy data): {type(e).__name__}", file=sys.stderr)
    finally:
        torch.cuda.empty_cache()

_prebuild()

# Force 1-stage for shapes where assembly kernel works
_orig = _fm.get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _force_1stage(token, model_dim, inter_dim, expert, topk, *rest_args, **kwargs):
    metadata = _orig(token, model_dim, inter_dim, expert, topk, *rest_args, **kwargs)
    if inter_dim <= 1024:
        metadata.run_1stage = True
        metadata.stage1 = functools.partial(
            _fm.fused_moe_1stage,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
        )
    return metadata

_fm.get_2stage_cfgs = _force_1stage


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
