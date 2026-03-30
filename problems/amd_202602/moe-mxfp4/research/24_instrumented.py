"""
Instrumented version to understand eval pipeline.
Prints call counts, shapes, and timing to stderr.
Uses version 23's optimizations underneath.
"""
import torch
import sys
import time
import functools
from task import input_t, output_t

from aiter import ActivationType, QuantType
import aiter.fused_moe as _fm

# Monkey-patch use_nt and block_m
_fm.use_nt = functools.lru_cache(maxsize=2048)(lambda token, topk, e: True)
_orig_get_block_size_M = _fm.get_block_size_M.__wrapped__
@functools.lru_cache(maxsize=2048)
def _custom_get_block_size_M(token, topk, expert, inter_dim):
    tokens_per_expert = (token * topk) / expert
    if tokens_per_expert < 64:
        return 32
    else:
        return _orig_get_block_size_M(token, topk, expert, inter_dim)
_fm.get_block_size_M = _custom_get_block_size_M

_call_count = {}
_first_call_time = {}


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

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    d_expert = config["d_expert"]

    key = (M, E, topk, d_expert)

    if key not in _call_count:
        _call_count[key] = 0
        _first_call_time[key] = time.perf_counter()

    _call_count[key] += 1

    # Log every 100 calls and at call 1, 2, 5, 10
    count = _call_count[key]
    if count in [1, 2, 5, 10, 50, 100, 500, 1000] or count % 1000 == 0:
        elapsed = (time.perf_counter() - _first_call_time[key]) * 1000
        print(f"[DIAG] shape=({M},{E},{topk},{d_expert}) call#{count} elapsed={elapsed:.1f}ms", file=sys.stderr)

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
