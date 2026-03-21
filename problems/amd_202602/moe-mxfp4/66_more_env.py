"""
v66: Additional ROCm environment variables for MI355X.
Building on v65's success with HIP_FORCE_DEV_KERNARG=1.
"""
import os
os.environ['HIP_FORCE_DEV_KERNARG'] = '1'
os.environ['GPU_MAX_HW_QUEUES'] = '2'
# Additional tuning
os.environ['HSA_ENABLE_SDMA'] = '0'  # Disable SDMA engine - forces GPU copy engine, can be faster for small transfers
os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '0'  # Disable tunable ops search overhead

import torch
from task import input_t, output_t

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
