#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#249: DEFINITIVE B-lane-to-column probe.
Each B lane gets a UNIQUE value (not l32%8 which aliases).
Use the first byte position only, with values that encode the lane uniquely.
A = uniform 1.0. Output[row][col] = sum * A_val * B_val_for_col.
By making A all 1.0 and B_lane_i have a unique non-zero in ONE position,
we can determine exactly which B-lane feeds which output column.

Strategy: B lane i has byte[0] = (i%8 + 1) | (((i/8)%8) << 4), all other bytes 0.
This gives each of 32 lanes a unique byte value.
For the MFMA with A=all 0x22 (FP4=1.0), scale=1.0:
output[any_row][col] = fp4_low(B_lane[col].byte[0]) * 1.0 + fp4_high(B_lane[col].byte[0]) * 1.0
                     = (low_nib_val + high_nib_val)
Since low=i%8 and high=i/8, output[r][c] encodes which B-lane maps to column c.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

_CONFIGS = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
def _inject():
    try: dev = arch_info.get_arch()
    except: dev = "gfx950"
    cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"; os.makedirs(cd, exist_ok=True)
    for sk, cfg in _CONFIGS.items():
        with open(f"{cd}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{sk}.json", "w") as f: json.dump(cfg, f)
try: _inject()
except: pass

_MODULE = 'probe249'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;

extern "C" __global__ __launch_bounds__(64)
void probe249(float* __restrict__ out) {
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;

    // A: uniform - all bytes 0x22 (FP4=1.0 in both nibbles)
    union { vi8 v; unsigned char b[32]; } ab;
    for (int i = 0; i < 8; i++) ab.v[i] = 0;
    for (int i = 0; i < 16; i++) ab.b[i] = 0x22;

    // B: unique per lane - only byte[0] is non-zero
    // Encode l32 as: low_nib = (l32 & 7) + 1, high_nib = (l32 >> 3) + 1
    // This gives l32=0: byte=0x11, l32=1: 0x12, ..., l32=7: 0x18,
    // l32=8: 0x21, l32=9: 0x22, ..., l32=31: 0x48
    // Only grp=0 has data (first 16 bytes = first 32 FP4), grp=1 all zeros
    union { vi8 v; unsigned char b[32]; } bb;
    for (int i = 0; i < 8; i++) bb.v[i] = 0;
    if (grp == 0) {
        unsigned char lo = (l32 & 7) + 1;  // 1-8
        unsigned char hi = (l32 >> 3) + 1; // 1-4
        bb.b[0] = lo | (hi << 4);
    }

    int scale = 127 | (127 << 8);
    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        ab.v, bb.v, acc, 4, 4, 0, scale, 0, scale);

    // Output
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = grp * 4 + i * 8 + j;
            int c = l32;
            out[r * 32 + c] = acc[i * 4 + j];
        }
}

torch::Tensor run_probe249() {
    auto out = torch::zeros({32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    hipLaunchKernelGGL(probe249, dim3(1), dim3(64), 0, 0, (float*)out.data_ptr());
    return out;
}
"""

_CPP = r"""
#include <torch/extension.h>
torch::Tensor run_probe249();
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP, cuda_sources=_HIP_SRC,
                       functions=['run_probe249'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
except Exception as e:
    print(f"[249] FAIL: {e}", file=sys.stderr)

if _mod:
    try:
        r = _mod.run_probe249().cpu().numpy()
        torch.cuda.synchronize()
        # FP4 values: 0=0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0, 8=0(neg)
        # For A=1.0 (both nibbles), B byte has lo=(l32&7)+1 and hi=(l32>>3)+1
        # MFMA computes: output = sum(A_fp4[k] * B_fp4[k]) over k=0..63
        # A has 64 FP4 values all = 1.0. B has 2 non-zero FP4 values (at k=0,1 from byte[0])
        # output[row][col] = 1.0 * fp4_val(lo) + 1.0 * fp4_val(hi)
        # where lo=(lane_for_col & 7)+1, hi=(lane_for_col >> 3)+1
        # fp4_val(1)=0.5, fp4_val(2)=1.0, fp4_val(3)=1.5, fp4_val(4)=2.0
        print("[249] Row 0:", [f"{r[0][c]:.2f}" for c in range(32)], file=sys.stderr)
        print("[249] Row 1:", [f"{r[1][c]:.2f}" for c in range(32)], file=sys.stderr)
        # Decode: for each col, output = fp4_val(lo) + fp4_val(hi)
        # Given the output, we can determine which l32 maps to which col
        fp4_vals = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        for c in range(32):
            v = r[0][c]
            # Find which (lo, hi) pair gives this value
            found = False
            for l in range(32):
                lo_idx = (l & 7) + 1
                hi_idx = (l >> 3) + 1
                expected = fp4_vals[lo_idx] + fp4_vals[hi_idx]
                if abs(v - expected) < 0.01:
                    if l != c:
                        print(f"[249] Col {c} <- B lane {l} (val={v:.2f}, expected lo={lo_idx} hi={hi_idx})", file=sys.stderr)
                    found = True
                    break
            if not found:
                print(f"[249] Col {c}: val={v:.2f} NO MATCH", file=sys.stderr)
    except Exception as e:
        print(f"[249] Error: {e}", file=sys.stderr)

_bc = [None, None, None]
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
