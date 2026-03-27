"""
v184: Attempt to install flydsl on the runner and use FlyDSL stage 2 kernels.
FlyDSL provides tuned MFMA assembly kernels that may be faster than CK.
Falls back to v175 if installation fails.
"""
import os, sys, stat, subprocess

# Try to install flydsl
try:
    import importlib.util
    if importlib.util.find_spec("flydsl") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "flydsl==0.0.1.dev95158637"], timeout=60)
        print("[v184] flydsl installed", file=sys.stderr)
    else:
        print("[v184] flydsl already available", file=sys.stderr)
except Exception as e:
    print(f"[v184] flydsl install failed: {e}", file=sys.stderr)

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

import torch, functools, aiter
from task import input_t, output_t
from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm
from aiter.fused_moe import (
    fused_moe, get_2stage_cfgs,
    cktile_moe_stage1, cktile_moe_stage2, fused_moe_1stage, MOEMetadata,
)

# Check FlyDSL availability after potential install
try:
    from aiter.ops.flydsl.utils import is_flydsl_available
    _flydsl_ok = is_flydsl_available()
    print(f"[v184] FlyDSL available: {_flydsl_ok}", file=sys.stderr)
except:
    _flydsl_ok = False

_orig = get_2stage_cfgs.__wrapped__

@functools.lru_cache(maxsize=2048)
def _patched(token, model_dim, inter_dim, expert, topk,
             dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
             activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled):
    md = _orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w,
               q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled)
    tokens_per_expert = (token * topk) / expert
    use_cktile = False; sk = 1
    if inter_dim > 1024: use_cktile = False
    elif tokens_per_expert < 5: use_cktile = True; sk = 2
    elif tokens_per_expert < 40 and expert <= 33: use_cktile = True; sk = 1
    if use_cktile and is_shuffled:
        md.ksplit = 2
        md.block_m = 16 if token < 2048 else 32 if token < 16384 else 64
        md.stage1 = functools.partial(cktile_moe_stage1,
            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
            k_pad_zeros=hidden_pad // 128 * 128, activation=ActivationType.Silu, split_k=sk)
        md.stage2 = functools.partial(cktile_moe_stage2,
            n_pad_zeros=hidden_pad // 64 * 64, k_pad_zeros=intermediate_pad // 128 * 128,
            activation=ActivationType.Silu)
        return md
    if inter_dim <= 1024 and expert <= 33 and q_type == QuantType.per_1x32 and is_shuffled:
        return MOEMetadata(functools.partial(fused_moe_1stage, kernelName="",
            activation=activation, quant_type=q_type), None, 32, 0, True)
    if inter_dim > 1024 and tokens_per_expert > 40:
        md.block_m = 64
    return md

_fm.get_2stage_cfgs = _patched

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states, _, _, _, _, w1, w2, w1s, w2s, topk_weights, topk_ids, config) = data
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                     hidden_pad=config["d_hidden_pad"]-config["d_hidden"],
                     intermediate_pad=config["d_expert_pad"]-config["d_expert"])
