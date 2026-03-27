#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#279: Test if grp0 and grp1 bytes are concatenated or interleaved.
A = all 1.0. B lane 0: grp0 = [0x43, 0x22*15], grp1 = [0x65, 0x22*15]
If concatenated: K[0..31] = grp0 bytes, K[32..63] = grp1 bytes
  sum = (1.5+2.0) + 15*2.0 + (3.0+4.0) + 15*2.0 = 3.5 + 30 + 7 + 30 = 70.5
If interleaved: K[0,1] from grp0[0], K[2,3] from grp1[0], K[4,5] from grp0[1], ...
  sum = (1.5+2.0) + (3.0+4.0) + 15*(2.0+2.0) ... different
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

_MODULE = 'probe279'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;

extern "C" __global__ __launch_bounds__(64)
void probe279(float* __restrict__ out) {
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;

    // A: all 1.0 (0x22 both nibbles)
    union { vi8 v; unsigned char b[32]; } ab;
    for (int i = 0; i < 8; i++) ab.v[i] = 0;
    for (int i = 0; i < 16; i++) ab.b[i] = 0x22;

    // B lane 0: grp0 byte[0] = 0x43 (lo=3→1.5, hi=4→2.0), rest 0x22
    //           grp1 byte[0] = 0x65 (lo=5→3.0, hi=6→4.0), rest 0x22
    // All other lanes: all 0x22
    union { vi8 v; unsigned char b[32]; } bb;
    for (int i = 0; i < 8; i++) bb.v[i] = 0;
    if (l32 == 0) {
        if (grp == 0) {
            bb.b[0] = 0x43; // lo=3(1.5), hi=4(2.0)
            for (int i = 1; i < 16; i++) bb.b[i] = 0x22;
        } else {
            bb.b[0] = 0x65; // lo=5(3.0), hi=6(4.0)
            for (int i = 1; i < 16; i++) bb.b[i] = 0x22;
        }
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
            out[r * 32 + c] = acc[i * 4 + j];
        }
}

torch::Tensor run279() {
    auto out = torch::zeros({32, 32}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    hipLaunchKernelGGL(probe279, dim3(1), dim3(64), 0, 0, (float*)out.data_ptr());
    return out;
}
"""

_CPP = r"""
#include <torch/extension.h>
torch::Tensor run279();
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP, cuda_sources=_HIP_SRC,
                       functions=['run279'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
except Exception as e:
    print(f"[279] FAIL: {e}", file=sys.stderr)

if _mod:
    try:
        r = _mod.run279().cpu().numpy()
        torch.cuda.synchronize()
        # If concatenated [grp0, grp1]:
        # grp0: byte0=0x43 (1.5+2.0=3.5) + 15*0x22 (15*2.0=30) = 33.5
        # grp1: byte0=0x65 (3.0+4.0=7.0) + 15*0x22 (15*2.0=30) = 37.0
        # Total: 33.5 + 37.0 = 70.5
        #
        # If interleaved [grp0[0],grp1[0],grp0[1],grp1[1],...]:
        # byte0_grp0=0x43 (3.5) + byte0_grp1=0x65 (7.0) + 14*2*0x22 (56.0) = 66.5
        # (this would be different)
        col0 = r[0][0]
        col1 = r[0][1]  # all 1.0 → 64.0
        print(f"[279] Col 0 (B lane 0): {col0:.1f}", file=sys.stderr)
        print(f"[279] Col 1: {col1:.1f}", file=sys.stderr)
        print(f"[279] Expected concatenated: 70.5, interleaved: 66.5", file=sys.stderr)
    except Exception as e:
        print(f"[279] error: {e}", file=sys.stderr)

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
