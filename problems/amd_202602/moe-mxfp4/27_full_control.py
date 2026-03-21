"""
Full control: monkey-patch get_2stage_cfgs to return hand-tuned configs
for each benchmark shape. Also set memory allocator config.

Key tuning knobs per shape:
- block_m: token grouping size
- use_non_temporal_load: L2 cache bypass
- kernel names: CK kernel variant selection
"""
import os
# Tune HIP memory allocator before any CUDA operations
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

import torch
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as _fm

# Store original function
_orig_get_2stage_cfgs = _fm.get_2stage_cfgs.__wrapped__

# Hand-tuned configs: (padded_token, model_dim, inter_dim, expert, topk) -> overrides
# Based on dsv3_fp4_tuned_fmoe.csv + our experiments
_HAND_TUNED = {
    # E=257, d_expert=256: use tuned CSV kernels + force use_nt=True
    # block_m from CSV, kernels from CSV
    (16, 7168, 256, 257, 9): {"block_m": 32, "use_nt": True},
    (64, 7168, 256, 257, 9): {"block_m": 32, "use_nt": True},
    (128, 7168, 256, 257, 9): {"block_m": 32, "use_nt": True},
    (256, 7168, 256, 257, 9): {"block_m": 32, "use_nt": True},
    (512, 7168, 256, 257, 9): {"block_m": 32, "use_nt": True},
    # E=33, d_expert=512: custom block_m + use_nt=True
    (16, 7168, 512, 33, 9): {"block_m": 32, "use_nt": True},
    (128, 7168, 512, 33, 9): {"block_m": 32, "use_nt": True},
    (512, 7168, 512, 33, 9): {"block_m": 64, "use_nt": True},
    # E=33, d_expert=2048: try block_m=64 instead of heuristic 128
    (512, 7168, 2048, 33, 9): {"block_m": 64, "use_nt": True},
}

@functools.lru_cache(maxsize=2048)
def _patched_get_2stage_cfgs(token, model_dim, inter_dim, expert, topk,
                              dtype, q_dtype_a, q_dtype_w, q_type,
                              use_g1u1, activation, doweight_stage1,
                              hidden_pad, intermediate_pad, is_shuffled=True):
    # Get original config first
    metadata = _orig_get_2stage_cfgs(
        token, model_dim, inter_dim, expert, topk,
        dtype, q_dtype_a, q_dtype_w, q_type,
        use_g1u1, activation, doweight_stage1,
        hidden_pad, intermediate_pad, is_shuffled,
    )

    # Apply our overrides
    key = (token, model_dim, inter_dim, expert, topk)
    overrides = _HAND_TUNED.get(key)
    if overrides:
        if "block_m" in overrides:
            metadata.block_m = overrides["block_m"]
        if "use_nt" in overrides:
            metadata.use_non_temporal_load = overrides["use_nt"]

    return metadata

_fm.get_2stage_cfgs = _patched_get_2stage_cfgs


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

    output = _fm.fused_moe(
        hidden_states,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None,
        a2_scale=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

    return output
