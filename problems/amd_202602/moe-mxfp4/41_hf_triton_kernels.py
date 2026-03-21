"""
Install triton_kernels from HuggingFace community build which has matmul_ogs.
Then try to use it for MoE GEMM.
"""
import subprocess
import sys

# Try installing from HF community build
try:
    from triton_kernels.matmul_ogs import matmul_ogs
except (ImportError, ModuleNotFoundError):
    print("[INSTALL] Installing triton_kernels from HF...", file=sys.stderr)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages",
             "triton_kernels", "--index-url", "https://huggingface.github.io/kernels-community/"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
        print("[INSTALL] Done", file=sys.stderr)
    except Exception as e:
        print(f"[INSTALL] HF install failed: {e}", file=sys.stderr)
        # Try pip install kernels (the OpenAI package name)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "kernels"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
            )
            print("[INSTALL] kernels package installed", file=sys.stderr)
        except Exception as e2:
            print(f"[INSTALL] kernels install also failed: {e2}", file=sys.stderr)

import torch
from task import input_t, output_t

# Check what's available now
_HAS_MATMUL_OGS = False
try:
    from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig
    from triton_kernels.routing import RoutingData, GatherIndx, ScatterIndx
    _HAS_MATMUL_OGS = True
    print("[CHECK] matmul_ogs available!", file=sys.stderr)
except ImportError as e:
    print(f"[CHECK] matmul_ogs not available: {e}", file=sys.stderr)

# Also check what triton_kernels contains
try:
    import triton_kernels
    print(f"[CHECK] triton_kernels dir: {[x for x in dir(triton_kernels) if not x.startswith('_')]}", file=sys.stderr)
except ImportError:
    print("[CHECK] triton_kernels not installed", file=sys.stderr)

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe


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
