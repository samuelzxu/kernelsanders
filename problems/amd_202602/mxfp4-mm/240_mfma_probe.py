#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #240: MFMA register layout probe.
Loads specific known patterns into A and B, runs MFMA, dumps output to stderr.
This reveals the exact register mapping without needing to match the reference.
Uses preshuffle for correctness, MFMA probe is diagnostic only.

Strategy: Load A where each row i has all bytes = i+1, B where each row j has
all bytes = j+1. After MFMA with scale=127 (1.0), the output[r][c] reveals
which A-row and B-row were multiplied. Since FP4 E2M1 with value byte 0x11
= (low=1=0.5, high=1=0.5), each byte contributes 0.5*0.5=0.25.
64 FP4 values per dot product = 32 bytes * 2 = 64 muls of 0.25? No...
Actually simpler: set all FP4 to same value, scale=1. output[r][c] = 64 * val_a * val_b.
But we need to identify WHICH row maps to WHICH output position.

Cleaner: Set A lane i's bytes all to pattern encoding i, B lane j's bytes encoding j.
Print the full 32x32 output. By inspecting output[r][c], we can determine:
- Which lane's A data maps to output row r
- Which lane's B data maps to output column c
"""
import os, shutil, sys
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

import json, torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

_CONFIGS = {
    "N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
    "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}},
}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(cd, exist_ok=True)
    for sk, cfg in _CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f:
            json.dump(cfg, f)
try: _inject()
except: pass

# MFMA probe kernel: loads synthetic data, runs 1 MFMA, dumps output
_MODULE = 'mfma_probe_240'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;

// Probe: each lane loads a UNIQUE pattern into A and B registers.
// A: lane l32 sets all 16 bytes to (l32+1). This means FP4 value = (l32+1)&0xF for both nibbles.
// B: lane l32 sets all 16 bytes to (l32+1). Same pattern.
// Scale = 127 (1.0) for all lanes.
// After MFMA, output[r][c] encodes which A-lane and B-lane combined.
// We dump the FULL 32x32 output to a float buffer.
extern "C" __global__ __launch_bounds__(64)
void mfma_probe(float* __restrict__ out) {
    int lane = threadIdx.x;
    int l32 = lane % 32;
    int grp = lane / 32;

    // A: each lane's row has FP4 pattern based on l32
    // Use byte value = ((l32 % 8) | ((l32 % 8) << 4)) so both nibbles = l32%8
    // FP4 values 0-7 are: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    union { vi8 v; unsigned char b[32]; } ab;
    for (int i = 0; i < 8; i++) ab.v[i] = 0;
    unsigned char a_val = (unsigned char)((l32 % 8) | ((l32 % 8) << 4));
    for (int i = 0; i < 16; i++) ab.b[i] = a_val;

    // B: same pattern but use different values to distinguish A vs B
    // B lane l32 uses value ((l32 % 8) + 1) % 8 to differentiate
    union { vi8 v; unsigned char b[32]; } bb;
    for (int i = 0; i < 8; i++) bb.v[i] = 0;
    // Actually, simpler: B lane l32 puts all bytes = 0x22 (both nibbles = 2, value = 1.0)
    // Then output = sum_k(A_fp4[k] * 1.0) * scale = sum of A's FP4 values
    // With 64 FP4 values per lane, output = 64 * fp4_val[l32%8]
    // This directly reveals which A-lane maps to each output position!
    for (int i = 0; i < 16; i++) bb.b[i] = 0x22; // both nibbles = 2 = FP4(1.0)

    // Scale = 127 = 2^0 = 1.0
    int scale_a = 127 | (127 << 8);
    int scale_b = 127 | (127 << 8);

    vf16 acc = {};
    for (int i = 0; i < 16; i++) acc[i] = 0.f;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        ab.v, bb.v, acc, 4, 4, 0, scale_a, 0, scale_b);

    // Store using known output mapping: acc[i*4+j] -> row=(grp*4+i*8+j), col=l32
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = grp * 4 + i * 8 + j;
            int c = l32;
            out[r * 32 + c] = acc[i * 4 + j];
        }
}

torch::Tensor run_probe() {
    auto out = torch::zeros({32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    hipLaunchKernelGGL(mfma_probe, dim3(1), dim3(64), 0, 0,
        (float*)out.data_ptr());
    return out;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run_probe();
"""

# Clear cache and compile
_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['run_probe'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
    print("[240] Probe compiled OK", file=sys.stderr)
except Exception as e:
    print(f"[240] Probe FAIL: {e}", file=sys.stderr)

# Run probe ONCE at import time, dump results
if _mod is not None:
    try:
        result = _mod.run_probe()
        torch.cuda.synchronize()
        r = result.cpu().numpy()
        # Print first few rows to understand the mapping
        # Each output[row][col] = 64 * fp4_val[A_lane_pattern]
        # fp4 values: 0=0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
        # So output / 64 gives the FP4 value, and we can map back to l32
        print(f"[240] PROBE OUTPUT (first 4 rows x 8 cols):", file=sys.stderr)
        for row in range(min(8, r.shape[0])):
            vals = [f"{r[row][c]:.1f}" for c in range(min(8, r.shape[1]))]
            print(f"  row {row}: {' '.join(vals)}", file=sys.stderr)
        # Print row 0, all 32 cols
        print(f"[240] Row 0 all cols: {[f'{r[0][c]:.1f}' for c in range(32)]}", file=sys.stderr)
        # Print col 0, all 32 rows
        print(f"[240] Col 0 all rows: {[f'{r[row][0]:.1f}' for row in range(32)]}", file=sys.stderr)
    except Exception as e:
        print(f"[240] Probe run error: {e}", file=sys.stderr)

# Actual submission: use preshuffle (proven correct)
_bc = [None, None, None]

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
