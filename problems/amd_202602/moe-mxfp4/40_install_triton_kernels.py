"""
Install triton_kernels at import time and use matmul_ogs for MoE GEMM.
This is a Triton-based grouped GEMM that handles MoE routing natively.
No CK assembly JIT needed - compiles in seconds via Triton.
"""
import subprocess
import sys

# Install triton_kernels from the triton repo
try:
    import triton_kernels
except ImportError:
    print("[INSTALL] Installing triton_kernels...", file=sys.stderr)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages",
         "git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels"],
        stdout=subprocess.DEVNULL,
    )
    print("[INSTALL] triton_kernels installed", file=sys.stderr)

import torch
from task import input_t, output_t

# Check if matmul_ogs is available after install
_USE_TRITON_MOE = False
try:
    from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig
    from triton_kernels.routing import routing, RoutingData, GatherIndx, ScatterIndx, compute_expt_data
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details.layout import StridedLayout
    from aiter.jit.utils.chip_info import get_gfx
    _USE_TRITON_MOE = True
    print("[CHECK] triton_kernels matmul_ogs available!", file=sys.stderr)
except ImportError as e:
    print(f"[CHECK] triton_kernels import failed: {e}", file=sys.stderr)

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

    # Fall back to fused_moe (triton_kernels integration is WIP)
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
