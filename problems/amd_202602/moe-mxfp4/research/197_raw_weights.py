"""
v197: Test using RAW (non-shuffled) weights with fused_moe.
If BNS kernels are available, this avoids preshuffle overhead.
The raw weights have is_shuffled=False, triggering different CK dispatch.
"""
import os, sys, stat
_JIT_DIR = "/home/runner/aiter/aiter/jit"
_BASE_URL = "https://github.com/samuelzxu/aiter-precompiled/releases/download/v0.3-rocm71"
_MODULES = ["module_aiter_enum.so","module_moe_sorting_opus.so","module_moe_sorting.so",
    "module_quant.so","module_activation.so","module_moe_cktile2stages.so",
    "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so",
    "module_moe_asm.so"]
def _install():
    import urllib.request
    os.makedirs(_JIT_DIR, exist_ok=True)
    for name in _MODULES:
        path = os.path.join(_JIT_DIR, name)
        if not os.path.exists(path):
            try: urllib.request.urlretrieve(f"{_BASE_URL}/{name}", path); os.chmod(path, 0o755)
            except Exception: pass
try: _install()
except Exception: pass

import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
os.environ['AITER_USE_NT'] = '1'

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states,
     gate_up_weight, down_weight,           # RAW weights
     gate_up_weight_scale, down_weight_scale,  # RAW scales
     w1_shuf, w2_shuf, w1s_shuf, w2s_shuf,   # Shuffled
     topk_weights, topk_ids, config) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Try using RAW weights (is_shuffled=False) — might trigger BNS kernel path
    try:
        output = fused_moe(hidden_states, gate_up_weight, down_weight,
                           topk_weights, topk_ids,
                           expert_mask=None, activation=ActivationType.Silu,
                           quant_type=QuantType.per_1x32, doweight_stage1=False,
                           w1_scale=gate_up_weight_scale, w2_scale=down_weight_scale,
                           a1_scale=None, a2_scale=None,
                           hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
        return output
    except Exception as e:
        print(f"[v197] Raw weights failed: {e}", file=sys.stderr)
        # Fallback to shuffled
        return fused_moe(hidden_states, w1_shuf, w2_shuf, topk_weights, topk_ids,
                         expert_mask=None, activation=ActivationType.Silu,
                         quant_type=QuantType.per_1x32, doweight_stage1=False,
                         w1_scale=w1s_shuf, w2_scale=w2s_shuf,
                         a1_scale=None, a2_scale=None,
                         hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
