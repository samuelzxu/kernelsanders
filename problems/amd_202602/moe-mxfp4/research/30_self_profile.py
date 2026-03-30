"""
Self-profiling version. On first call, profiles the kernel and prints
the GPU kernel breakdown to stderr. Uses v28 underneath.
"""
import torch
import sys
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_profiled = set()

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
    d_expert = config["d_expert"]
    key = (M, E, d_expert)

    def _run():
        return fused_moe(
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

    # Profile on second call (first call has JIT overhead)
    if key not in _profiled:
        _profiled.add(key)
        # Warmup
        result = _run()
        # Profile
        try:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                result = _run()
                torch.cuda.synchronize()
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)
            print(f"\n[PROFILE] shape=({M},{E},{d_expert})\n{table}", file=sys.stderr)
        except Exception as e:
            print(f"[PROFILE ERROR] {e}", file=sys.stderr)
            result = _run()
        return result

    return _run()
