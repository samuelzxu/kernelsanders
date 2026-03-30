"""
Check if triton_kernels (GPT-OSS) is available on the runner.
If so, use matmul_ogs for MoE GEMM instead of AITER CK kernels.
"""
import sys
import torch
from task import input_t, output_t

# Check available packages
_HAS_TRITON_KERNELS = False
try:
    from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig
    from triton_kernels.routing import routing
    _HAS_TRITON_KERNELS = True
    print("[CHECK] triton_kernels available!", file=sys.stderr)
except ImportError as e:
    print(f"[CHECK] triton_kernels NOT available: {e}", file=sys.stderr)

# Check ATOM
_HAS_ATOM = False
try:
    import atom
    _HAS_ATOM = True
    print("[CHECK] ATOM available!", file=sys.stderr)
except ImportError as e:
    print(f"[CHECK] ATOM NOT available: {e}", file=sys.stderr)

# List relevant installed packages
try:
    import importlib.metadata
    for pkg in ['triton-kernels', 'triton_kernels', 'gpt-oss', 'atom', 'aiter']:
        try:
            ver = importlib.metadata.version(pkg)
            print(f"[CHECK] {pkg}=={ver}", file=sys.stderr)
        except importlib.metadata.PackageNotFoundError:
            pass
except Exception:
    pass

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
