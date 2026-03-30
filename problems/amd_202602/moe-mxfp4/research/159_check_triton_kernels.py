"""
v159: Check if triton_kernels (matmul_ogs) is available on runner.
Also check Triton version and capabilities.
"""
import os, sys, stat
_JIT_DIR = "/home/runner/aiter/aiter/jit"
_BASE_URL = "https://github.com/samuelzxu/aiter-precompiled/releases/download/v0.3-rocm71"
_MODULES = ["module_aiter_enum.so","module_moe_sorting_opus.so","module_moe_sorting.so",
    "module_quant.so","module_activation.so","module_moe_cktile2stages.so",
    "module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_.so"]
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

# Check Triton version and matmul_ogs availability
import triton
print(f"[v159] Triton version: {triton.__version__}", file=sys.stderr)
print(f"[v159] Triton path: {triton.__file__}", file=sys.stderr)

try:
    from triton_kernels.matmul_ogs import matmul_ogs
    print("[v159] triton_kernels.matmul_ogs: AVAILABLE!", file=sys.stderr)
except ImportError as e:
    print(f"[v159] triton_kernels.matmul_ogs: NOT available ({e})", file=sys.stderr)

try:
    import triton_kernels
    print(f"[v159] triton_kernels: {triton_kernels.__file__}", file=sys.stderr)
except ImportError:
    print("[v159] triton_kernels: NOT installed", file=sys.stderr)
    # Try pip install
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "install", "triton-kernels", "-q"],
                          capture_output=True, text=True, timeout=60)
    print(f"[v159] pip install triton-kernels: {result.returncode}", file=sys.stderr)
    if result.returncode == 0:
        try:
            from triton_kernels.matmul_ogs import matmul_ogs
            print("[v159] After install, matmul_ogs: AVAILABLE!", file=sys.stderr)
        except ImportError as e2:
            print(f"[v159] After install, still not available: {e2}", file=sys.stderr)

# Fallback to CK
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    (hidden_states, _, _, _, _, w1, w2, w1s, w2s, topk_weights, topk_ids, config) = data
    return fused_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1s, w2_scale=w2s, a1_scale=None, a2_scale=None,
                     hidden_pad=config["d_hidden_pad"]-config["d_hidden"],
                     intermediate_pad=config["d_expert_pad"]-config["d_expert"])
