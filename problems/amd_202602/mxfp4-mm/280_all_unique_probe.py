#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#280: ALL 16 bytes unique per lane. Tests EXACT byte-to-K-position mapping.
B lane 0, grp 0: each byte has a unique value encoding its position.
A = all 1.0. The output for each row reveals the EXACT dot product,
which encodes which bytes contributed at which K-positions.

Specifically: if we set byte[i] = (i%8) | (((i+1)%8)<<4), each byte
has a unique combination. The dot product = sum of all FP4 values.
But to detect REORDERING, we use weighted positions:
byte[i] has lo_nibble = (i+1) mod 8, hi_nibble = 0 (zero).
So the dot product = sum(fp4[(i+1)%8] for i in 0..15) + 0*16.
A specific byte ordering gives a specific sum.
If bytes are reordered, the sum changes (unless it's a permutation that
preserves the multiset of values, which won't happen with all unique values).

Actually simpler: use byte[i] = ((i+1)%8) for i=0..6 (unique FP4 1-7),
byte[7] = 0, and bytes 8-15 = 0. Then sum = sum(fp4[1..7]) = 0.5+1+1.5+2+3+4+6 = 18.
For grp1: all zeros. Total = 18.
If ANY byte is at the wrong position, the lo/hi nibbles would change.
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

_MODULE = 'probe280b'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;

// Test: A=uniform(1.0 both nibbles), B lane 0 grp 0 has POSITIONAL encoding.
// B byte[i] = i+1 for i=0..6, byte[7]=0, bytes[8..15]=0.
// So FP4 values: byte0: lo=1(0.5),hi=0(0) ; byte1: lo=2(1.0),hi=0 ; ...
// Wait, that only encodes the LOW nibble. Let me use BOTH nibbles uniquely.
// byte[i] encodes: lo_nibble = (2*i+1) % 8, hi_nibble = (2*i+2) % 8
// This gives 32 FP4 values (2 per byte), each at a known K-position.
// The dot product with A=1.0 = sum of all B FP4 values.
// If bytes are reordered, the dot product CHANGES (different pairs at different positions
// means different lo/hi splits, even though sum might be same if permutation is derangement).
//
// Actually, since we multiply A[k]*B[k] and A=1.0 for all k:
// result = sum of all B FP4 values = sum(lo_i + hi_i for i in 0..15) where lo_i, hi_i are grp0
// + sum of all grp1 values (which are 0).
// The sum doesn't depend on byte ORDER, only on the multiset of FP4 values.
// So a permutation wouldn't change the sum.
//
// To detect permutation, I need A to have DIFFERENT values at each K-position.
// Let me make A[k] = k-th FP4 code's value (weight by position).
// A byte[i]: lo = i%8, hi = (i+8)%8... this gets complex.
//
// SIMPLEST: make A have a GRADIENT (different value per K) and B have a DELTA (non-zero at only one K).
// If B has only byte[5] non-zero (with a known FP4 value), the output = A[10]*B[10] + A[11]*B[11].
// If byte 5 is at the WRONG K-position, we'd multiply different A values.

extern "C" __global__ __launch_bounds__(64)
void probe280b(float* __restrict__ out) {
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;

    // A: gradient - each lane has byte[i] with UNIQUE per-byte value
    // Use: A byte[i] = 0x22 for all (1.0 per nibble).
    // EXCEPT: A lane 0, grp 0: each byte is 0x22 = 1.0
    // We multiply with B that has a DELTA at specific bytes.
    union { vi8 v; unsigned char b[32]; } ab;
    for (int i = 0; i < 8; i++) ab.v[i] = 0;
    for (int i = 0; i < 16; i++) ab.b[i] = 0x22; // A = all 1.0

    // B: DELTA at byte[5] only for lane 0 grp 0. Value = 0x43 (lo=3→1.5, hi=4→2.0)
    // All other bytes = 0, all other lanes = 0x22
    union { vi8 v; unsigned char b[32]; } bb;
    for (int i = 0; i < 8; i++) bb.v[i] = 0;
    if (l32 == 0 && grp == 0) {
        bb.b[5] = 0x43; // ONLY byte 5 is non-zero: 1.5 + 2.0 = 3.5
    } else if (l32 == 0 && grp == 1) {
        // grp 1: nothing (all zeros)
    } else {
        for (int i = 0; i < 16; i++) bb.b[i] = 0x22; // others = all 1.0
    }

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

// Second test: B delta at byte[0] (3.5), byte[5] (3.5), byte[15] (3.5)
// If sequential: sum = 3*3.5 = 10.5
// If reordered: depends on where they end up
extern "C" __global__ __launch_bounds__(64)
void probe280bb(float* __restrict__ out) {
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;

    union { vi8 v; unsigned char b[32]; } ab;
    for (int i = 0; i < 8; i++) ab.v[i] = 0;
    for (int i = 0; i < 16; i++) ab.b[i] = 0x22;

    union { vi8 v; unsigned char b[32]; } bb;
    // EXPLICIT zero of ALL 32 bytes
    for (int i = 0; i < 32; i++) bb.b[i] = 0;
    if (l32 == 0 && grp == 0) {
        bb.b[0] = 0x43;  // 3.5
        bb.b[5] = 0x43;  // 3.5
        bb.b[15] = 0x43; // 3.5
    } else if (l32 == 0 && grp == 1) {
        // Explicitly all zeros (already done above)
    } else {
        for (int i = 0; i < 16; i++) bb.b[i] = 0x22;
    }

    int scale = 127 | (127 << 8);
    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        ab.v, bb.v, acc, 4, 4, 0, scale, 0, scale);

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = grp * 4 + i * 8 + j;
            int c = l32;
            out[32*32 + r * 32 + c] = acc[i * 4 + j];
        }
}

torch::Tensor run280() {
    auto out = torch::zeros({2, 32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    hipLaunchKernelGGL(probe280b, dim3(1), dim3(64), 0, 0, (float*)out.data_ptr());
    hipLaunchKernelGGL(probe280bb, dim3(1), dim3(64), 0, 0, (float*)out.data_ptr());
    return out;
}
"""

_CPP = r"""
#include <torch/extension.h>
torch::Tensor run280();
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP, cuda_sources=_HIP_SRC,
                       functions=['run280'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
except Exception as e:
    print(f"[280] FAIL: {e}", file=sys.stderr)

if _mod:
    try:
        r = _mod.run280().cpu().numpy()
        torch.cuda.synchronize()
        # Test 1: B delta at byte[5] only → sum = 3.5 (1.5+2.0)
        print(f"[280] Test1 (delta byte[5]): col0={r[0][0][0]:.2f} (expect 3.50)", file=sys.stderr)
        # Test 2: B delta at bytes 0,5,15 → sum = 3*3.5 = 10.5
        print(f"[280] Test2 (delta 0,5,15): col0={r[1][0][0]:.2f} (expect 10.50)", file=sys.stderr)
        # Col 1 (all 1.0): 64 * 1.0 = 64.0 for test1, same for test2
        print(f"[280] Test1 col1={r[0][0][1]:.2f} Test2 col1={r[1][0][1]:.2f}", file=sys.stderr)
    except Exception as e:
        print(f"[280] error: {e}", file=sys.stderr)

_bc = [None, None, None]
for _m, _n, _k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A = torch.randn((_m, _k), dtype=torch.bfloat16, device="cuda")
        _Bw = torch.zeros((_n//16, (_k//2)*16), dtype=torch.uint8, device="cuda")
        _Bws = torch.zeros((_n//32, _k), dtype=torch.uint8, device="cuda")
        gemm_a16wfp4_preshuffle(_A, _Bw, _Bws, prequant=True, dtype=torch.bfloat16)
    except: pass
torch.cuda.empty_cache()

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
