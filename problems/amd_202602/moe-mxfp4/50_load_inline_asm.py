"""
Load pre-compiled assembly MoE kernel via torch.utils.cpp_extension.load_inline.
C++ code loads the .co file using hipModuleLoad and launches it.
Compilation happens at import time (before benchmarking).
"""
import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

# Find the .co file path
import aiter as _aiter
_aiter_root = os.path.dirname(os.path.dirname(_aiter.__file__))
_co_path = os.path.join(_aiter_root, "hsa", "gfx950", "fmoe", "silu",
                        "fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512.co")

# C++ source for loading and launching the assembly kernel
_cpp_source = f'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
// stdio for output

static hipModule_t g_module = nullptr;
static hipFunction_t g_kernel = nullptr;
static bool g_loaded = false;

// Packed kernel args struct matching asm_fmoe.cu layout
// Each pointer is followed by 8 bytes padding (p2)
// Each uint32 is followed by 12 bytes padding (p3)
// Total: 12 pointers * 16 + 16 uint32s * 16 = 448 bytes

bool load_asm_kernel() {{
    if (g_loaded) return true;

    const char* co_path = "{_co_path}";
    FILE* fp = fopen(co_path, "rb");
    if (!fp) {{
        fprintf(stderr, "[ASM] Cannot open: %s\\n", co_path);
        return false;
    }}
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> data(fsize);
    fread(data.data(), 1, fsize, fp);
    fclose(fp);

    hipError_t err = hipModuleLoadData(&g_module, data.data());
    if (err != hipSuccess) {{
        fprintf(stderr, "[ASM] hipModuleLoadData failed: %s\\n", hipGetErrorString(err));
        return false;
    }}

    const char* kernel_name = "_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512E";
    err = hipModuleGetFunction(&g_kernel, g_module, kernel_name);
    if (err != hipSuccess) {{
        fprintf(stderr, "[ASM] hipModuleGetFunction failed: %s\\n", hipGetErrorString(err));
        return false;
    }}

    fprintf(stderr, "[ASM] Kernel loaded successfully\\n");
    g_loaded = true;
    return true;
}}

bool is_asm_loaded() {{
    return g_loaded;
}}
'''

_cpp_functions = '''
bool load_asm_kernel();
bool is_asm_loaded();
'''

# Compile the C++ extension at import time
try:
    _asm_module = load_inline(
        name='asm_moe_launcher',
        cpp_sources=_cpp_functions,
        cuda_sources=_cpp_source,
        functions=['load_asm_kernel', 'is_asm_loaded'],
        verbose=False,
        extra_cflags=['-O2'],
        extra_cuda_cflags=['-O2'],
    )
    # Try loading the kernel
    _asm_loaded = _asm_module.load_asm_kernel()
    import sys
    print(f"[ASM] load_asm_kernel returned: {_asm_loaded}", file=sys.stderr)
except Exception as e:
    import sys
    print(f"[ASM] load_inline failed: {e}", file=sys.stderr)
    _asm_loaded = False


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

    # For now, always use fused_moe (ASM launch integration WIP)
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
