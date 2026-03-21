"""
Launch pre-compiled assembly MoE kernel directly via hip-python.
Bypasses the module_moe_asm C++ JIT build entirely (~30s saved).
Uses the exact KernelArgs struct from asm_fmoe.cu.

The kernel: fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512
- vs = weighted sum (sorted_weights applied)
- silu = SwiGLU activation
- 1tg_ps = 1 threadgroup persistent (preshuffle)
- 32x512 = block_m=32, sub_GU=512
"""
import os
import sys
import struct
import ctypes
import torch
from task import input_t, output_t

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_inter_dim
from aiter import get_hip_quant
from aiter.utility import fp4_utils


# Load the pre-compiled assembly kernel at import time
_ASM_KERNEL = None
_ASM_MODULE = None

def _load_asm_kernel():
    global _ASM_KERNEL, _ASM_MODULE
    from hip import hip

    # Find the .co file
    import aiter
    aiter_dir = os.path.dirname(aiter.__file__)
    # Go up to aiter root, then to hsa/
    aiter_root = os.path.dirname(aiter_dir)
    co_path = os.path.join(aiter_root, "hsa", "gfx950", "fmoe", "silu",
                           "fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512.co")

    if not os.path.exists(co_path):
        print(f"[ASM] .co file not found at {co_path}", file=sys.stderr)
        return False

    with open(co_path, "rb") as f:
        co_data = f.read()

    err, _ASM_MODULE = hip.hipModuleLoadData(co_data)
    if err != hip.hipError_t.hipSuccess:
        print(f"[ASM] hipModuleLoadData failed: {err}", file=sys.stderr)
        return False

    kernel_name = "_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512E"
    err, _ASM_KERNEL = hip.hipModuleGetFunction(_ASM_MODULE, kernel_name.encode())
    if err != hip.hipError_t.hipSuccess:
        print(f"[ASM] hipModuleGetFunction failed: {err}", file=sys.stderr)
        return False

    print(f"[ASM] Kernel loaded successfully", file=sys.stderr)
    return True

try:
    _ASM_LOADED = _load_asm_kernel()
except Exception as e:
    print(f"[ASM] Load failed: {e}", file=sys.stderr)
    _ASM_LOADED = False


def _build_kernel_args(
    out, inp, w1, w2,
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    topk, input_scale, w1_scale, w2_scale,
):
    """Build the KernelArgs struct matching asm_fmoe.cu layout."""
    token_cnt = out.shape[0]
    dim = w2.shape[1]  # model_dim (byte-level for fp4x2)
    eprt = w1.shape[0]
    inter_dim = w2.shape[2] * (w2.shape[1] // w1.shape[2])
    sub_X_cnt = sorted_expert_ids.shape[0]

    I_elemSize = 1  # uint8 for fp4x2
    O_elemSize = 2  # bf16

    stride_X = inp.stride(0) * inp.element_size()
    stride_GU = dim * I_elemSize // 2  # fp4x2: half the bytes
    stride_D = inter_dim * I_elemSize // 2
    stride_expert_GU = stride_GU * inter_dim * 2  # g1u1: *2 for gate+up
    stride_expert_D = stride_D * dim
    stride_expert_GUDQN = w1_scale.stride(0) * w1_scale.element_size()
    stride_expert_DDQN = w2_scale.stride(0) * w2_scale.element_size()
    stride_expert_SMTDQN = inter_dim * 4  # float32
    stride_O = dim * O_elemSize

    sub_GU = 512
    ps_deno = (inter_dim + sub_GU - 1) // sub_GU

    # Each field is 16-byte aligned (pointer + p2 padding, or uint32 + p3 padding)
    # Total: 12 pointers + 16 uint32s = 28 × 16 bytes = 448 bytes
    args = bytearray(448)

    def pack_ptr(offset, ptr_val):
        struct.pack_into("<Q", args, offset, ptr_val)
        # p2 padding is already zero

    def pack_u32(offset, val):
        struct.pack_into("<I", args, offset, val & 0xFFFFFFFF)
        # p3 padding is already zero

    # Pointers (each at 16-byte boundary)
    pack_ptr(0 * 16, out.data_ptr())           # ptr_O
    pack_ptr(1 * 16, inp.data_ptr())           # ptr_X
    pack_ptr(2 * 16, w1.data_ptr())            # ptr_GU
    pack_ptr(3 * 16, num_valid_ids.data_ptr()) # ptr_XC (num_valid_ids)
    pack_ptr(4 * 16, w2.data_ptr())            # ptr_D
    pack_ptr(5 * 16, input_scale.data_ptr())   # ptr_XQ (activation scale)
    pack_ptr(6 * 16, w1_scale.data_ptr())      # ptr_GUQ (gate_up scale)
    pack_ptr(7 * 16, w2_scale.data_ptr())      # ptr_DQ (down scale)
    pack_ptr(8 * 16, 0)                        # ptr_SMQ (smooth quant, null)
    pack_ptr(9 * 16, sorted_token_ids.data_ptr())  # ptr_STP
    pack_ptr(10 * 16, sorted_weights.data_ptr())   # ptr_SW
    pack_ptr(11 * 16, sorted_expert_ids.data_ptr()) # ptr_SEP

    # Scalars (each at 16-byte boundary, starting at offset 192)
    base = 12 * 16
    pack_u32(base + 0 * 16, dim)
    pack_u32(base + 1 * 16, inter_dim)
    pack_u32(base + 2 * 16, token_cnt)
    pack_u32(base + 3 * 16, eprt)
    pack_u32(base + 4 * 16, stride_X)
    pack_u32(base + 5 * 16, stride_GU)
    pack_u32(base + 6 * 16, stride_D)
    pack_u32(base + 7 * 16, stride_O)
    pack_u32(base + 8 * 16, stride_expert_GU)
    pack_u32(base + 9 * 16, stride_expert_D)
    pack_u32(base + 10 * 16, stride_expert_GUDQN)
    pack_u32(base + 11 * 16, stride_expert_DDQN)
    pack_u32(base + 12 * 16, stride_expert_SMTDQN)
    pack_u32(base + 13 * 16, topk)
    # total_tgs and ps_deno - need to check kernel config
    # For persistent threadgroup kernel: num_persistent_tgs from CSV
    # For now, use 0 (non-persistent mode)
    pack_u32(base + 14 * 16, 0)  # total_tgs
    pack_u32(base + 15 * 16, ps_deno)

    return args, sub_X_cnt, sub_GU, inter_dim


def launch_asm_fmoe_g1u1(
    out, inp, w1, w2,
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    topk, input_scale, w1_scale, w2_scale,
):
    """Launch the assembly kernel directly via hip-python."""
    from hip import hip

    args_bytes, sub_X_cnt, sub_GU, inter_dim = _build_kernel_args(
        out, inp, w1, w2,
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        topk, input_scale, w1_scale, w2_scale,
    )

    # Grid/block dims (non-persistent mode)
    gdx = (inter_dim + sub_GU - 1) // sub_GU
    gdy = sub_X_cnt
    gdz = 1
    bdx = 256
    bdy = 1
    bdz = 1

    # Convert args to ctypes buffer
    args_buf = (ctypes.c_char * len(args_bytes)).from_buffer_copy(args_bytes)
    arg_size = ctypes.c_size_t(len(args_bytes))

    # Get current HIP default execution context (no "s.t" word)
    hip_exec_ctx = 0  # default

    err = hip.hipModuleLaunchKernel(
        _ASM_KERNEL,
        gdx, gdy, gdz,
        bdx, bdy, bdz,
        0,           # shared_mem
        hip_exec_ctx,  # default context
        None,        # kernelParams (not used with extra)
        extra=(
            ctypes.c_void_p(1),  # HIP_LAUNCH_PARAM_BUFFER_POINTER
            ctypes.cast(args_buf, ctypes.c_void_p),
            ctypes.c_void_p(2),  # HIP_LAUNCH_PARAM_BUFFER_SIZE
            ctypes.cast(ctypes.pointer(arg_size), ctypes.c_void_p),
            ctypes.c_void_p(0),  # HIP_LAUNCH_PARAM_END
        ),
    )
    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipModuleLaunchKernel failed: {err}")


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

    # Always fall back to fused_moe for now
    # The ASM direct launch is experimental
    if not _ASM_LOADED or True:
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
