#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#274: Intra-lane byte ordering probe.
Each lane puts a DIFFERENT FP4 value at each of its 16 byte positions.
A = uniform 1.0. B has lane 0 with byte[i] = i+1 (values 0.5 to 6.0).
Output = sum over k of A[k] * B[k] = sum of B's FP4 values (since A=1.0)
If bytes are sequential in K: output = sum(fp4[byte[0]_lo], fp4[byte[0]_hi], fp4[byte[1]_lo], ...)
If bytes are non-sequential: different sum.
By checking the sum, we can verify byte ordering.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

_MODULE = 'probe274'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;

// FP4 E2M1 values for codes 0-7: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Byte 0x12 = lo=2(1.0), hi=1(0.5) → contributes 1.0 + 0.5 = 1.5 per byte
// Byte 0x34 = lo=4(2.0), hi=3(1.5) → contributes 2.0 + 1.5 = 3.5

extern "C" __global__ __launch_bounds__(64)
void probe274(float* __restrict__ out) {
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;

    // A: all lanes = all bytes 0x22 (both nibbles = 2 = FP4(1.0))
    union { vi8 v; unsigned char b[32]; } ab;
    for (int i = 0; i < 8; i++) ab.v[i] = 0;
    for (int i = 0; i < 16; i++) ab.b[i] = 0x22;

    // B: lane 0 has unique bytes, all other lanes = all 0x22 (1.0)
    // Lane 0, grp 0: byte[i] = i+1 (gives different FP4 values per position)
    // Lane 0, grp 1: byte[i] = 0x22 (all 1.0)
    // All other lanes: byte[i] = 0x22 (all 1.0)
    union { vi8 v; unsigned char b[32]; } bb;
    for (int i = 0; i < 8; i++) bb.v[i] = 0;
    if (l32 == 0 && grp == 0) {
        // Each byte encodes 2 FP4 values. Use a known sequence:
        // byte[0] = 0x21 → lo=1(0.5), hi=2(1.0) → sum=1.5
        // byte[1] = 0x43 → lo=3(1.5), hi=4(2.0) → sum=3.5
        // byte[2] = 0x65 → lo=5(3.0), hi=6(4.0) → sum=7.0
        // byte[3] = 0x07 → lo=7(6.0), hi=0(0.0) → sum=6.0
        // bytes[4..15] = 0x22 → each contributes 2.0
        bb.b[0] = 0x21; bb.b[1] = 0x43; bb.b[2] = 0x65; bb.b[3] = 0x07;
        for (int i = 4; i < 16; i++) bb.b[i] = 0x22;
        // Expected sum if sequential: 1.5 + 3.5 + 7.0 + 6.0 + 12*2.0 = 42.0
    } else {
        for (int i = 0; i < 16; i++) bb.b[i] = 0x22;
        // All 1.0: sum = 32 * 1.0 = 32.0
    }

    // Scale = 127 (1.0) for all
    int scale = 127 | (127 << 8);
    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        ab.v, bb.v, acc, 4, 4, 0, scale, 0, scale);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = grp * 4 + i * 8 + j;
            int c = l32;
            out[r * 32 + c] = acc[i * 4 + j];
        }
}

torch::Tensor run274() {
    auto out = torch::zeros({32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    hipLaunchKernelGGL(probe274, dim3(1), dim3(64), 0, 0, (float*)out.data_ptr());
    return out;
}
"""

_CPP = r"""
#include <torch/extension.h>
torch::Tensor run274();
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP, cuda_sources=_HIP_SRC,
                       functions=['run274'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
except Exception as e:
    print(f"[274] FAIL: {e}", file=sys.stderr)

if _mod:
    try:
        r = _mod.run274().cpu().numpy()
        torch.cuda.synchronize()
        # Col 0 = B lane 0. If bytes are sequential:
        # grp0 bytes: 0x21,0x43,0x65,0x07 + 12*0x22
        # → FP4 sum from grp0: 0.5+1.0+1.5+2.0+3.0+4.0+6.0+0.0 + 12*2.0 = 42.0
        # grp1 bytes: all 0x22 → 32 * 1.0 = 32.0
        # Total: 42.0 + 32.0 = 74.0
        # But with A = all 1.0: output = sum_k(1.0 * B[k]) = sum(B's FP4 values)
        print(f"[274] Col 0 (B lane 0): {[f'{r[row][0]:.1f}' for row in range(8)]}", file=sys.stderr)
        print(f"[274] Col 1 (B lane 1, all 1.0): {[f'{r[row][1]:.1f}' for row in range(4)]}", file=sys.stderr)
        # Expected col 0: all rows same = 74.0 (42.0 from grp0 + 32.0 from grp1)
        # Expected col 1: all rows same = 64.0 (32 * 1.0 from each group = 32+32)
        print(f"[274] Expected col0=74.0 col1=64.0", file=sys.stderr)
    except Exception as e:
        print(f"[274] error: {e}", file=sys.stderr)

_bc = [None, None, None]
@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A = data[0]; B_shuffle = data[3]; B_scale_sh = data[4]
    m, k = A.shape; n = data[1].shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _bc[0]:
        _bc[0] = dp
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
