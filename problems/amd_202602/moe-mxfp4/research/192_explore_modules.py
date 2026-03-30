"""
v192: Diagnostic — explore what CK modules and kernels are available on the runner.
List all .so files in the AITER jit dir and available FlyDSL kernel configs.
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

import torch, functools, aiter, glob
from task import input_t, output_t
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe

# Explore available modules and kernels
print("[v192] Exploring runner capabilities:", file=sys.stderr)

# List JIT modules
jit_dir = "/home/runner/aiter/aiter/jit"
for f in sorted(glob.glob(os.path.join(jit_dir, "*.so"))):
    print(f"  JIT module: {os.path.basename(f)}", file=sys.stderr)

# List ASM kernels
asm_dir = "/home/runner/aiter/hsa/gfx950/fmoe/silu"
if os.path.exists(asm_dir):
    for f in sorted(os.listdir(asm_dir)):
        print(f"  ASM kernel: {f}", file=sys.stderr)

# Check FlyDSL kernel registry
try:
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params
    # Test various FlyDSL kernel names
    test_names = [
        "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_reduce",
        "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce",
        "flydsl_moe2_afp4_wfp4_bf16_t128x256x256_reduce",
        "flydsl_moe2_afp4_wfp4_bf16_t32x128x128_atomic",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x128_atomic",
        "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic",
        "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_atomic",
        "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce",
        "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_reduce",
        "flydsl_moe1_afp4_wfp4_bf16_t32x256x256",
        "flydsl_moe1_afp4_wfp4_bf16_t64x256x256",
    ]
    for name in test_names:
        try:
            params = get_flydsl_kernel_params(name)
            if params:
                print(f"  FlyDSL valid: {name} -> {params}", file=sys.stderr)
            else:
                print(f"  FlyDSL invalid: {name}", file=sys.stderr)
        except Exception as e:
            print(f"  FlyDSL error: {name} -> {e}", file=sys.stderr)
except Exception as e:
    print(f"  FlyDSL registry error: {e}", file=sys.stderr)

# Check available CK module functions
try:
    import aiter
    ck_funcs = [f for f in dir(aiter) if 'moe' in f.lower() or 'ck' in f.lower()]
    for f in sorted(ck_funcs):
        print(f"  CK func: {f}", file=sys.stderr)
except Exception as e:
    print(f"  CK func error: {e}", file=sys.stderr)

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states, _, _, _, _, w1, w2, w1s, w2s, topk_weights, topk_ids, config) = data
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                     hidden_pad=config["d_hidden_pad"]-config["d_hidden"],
                     intermediate_pad=config["d_expert_pad"]-config["d_expert"])
