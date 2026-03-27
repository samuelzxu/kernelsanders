#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#278: HIP kernel that dumps its computed A FP4 bytes for row 0.
Compare with dynamic_mxfp4_quant output to find the quant bug.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

_MODULE = 'dump278'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>
typedef hip_bfloat16 bf16;

// Same sw_quant as #272
__device__ __forceinline__ unsigned char sw_fp4(float val, float qscale) {
    float qx = val * qscale;
    unsigned int bits = __float_as_uint(qx);
    unsigned int s = bits & 0x80000000u;
    bits ^= s;
    float ax = __uint_as_float(bits);
    unsigned char r;
    if (ax >= 6.0f) r = 0x7;
    else if (ax < 1.0f) {
        float dx = ax + __uint_as_float(0x4A800000u);
        unsigned int db = __float_as_uint(dx) - 0x4A800000u;
        r = (unsigned char)(db & 0xFF);
    } else {
        unsigned int nx = bits;
        unsigned int mo = (nx >> 22) & 1;
        nx += (unsigned int)((int)((1-127)<<23) + (1<<21) - 1);
        nx += mo;
        nx >>= 22;
        r = (unsigned char)(nx & 0xFF);
    }
    r |= (unsigned char)(s >> 28);
    return r;
}

// Dump A FP4 bytes for row 0, first 32 bf16 values → 16 packed FP4 bytes
extern "C" __global__ void dump_quant(
    const bf16* __restrict__ A,  // (M, K) bf16
    unsigned char* __restrict__ fp4_out,  // (16,) dumped FP4 bytes
    unsigned char* __restrict__ scale_out, // (1,) E8M0 scale
    int K
) {
    // Only thread 0 does anything
    if (threadIdx.x != 0) return;

    // Read first 32 bf16 values of row 0
    float vals[32];
    float mx = 0.f;
    for (int i = 0; i < 32 && i < K; i++) {
        vals[i] = static_cast<float>(A[i]);
        float av = fabsf(vals[i]);
        if (av > mx) mx = av;
    }

    // Compute E8M0 scale
    unsigned char e8m0 = 0u;
    float qs = 1.0f;
    if (mx > 0.f) {
        unsigned int ai = __float_as_uint(mx);
        ai = (ai + 0x200000u) & 0xFF800000u;
        int be = (int)((ai >> 23) & 0xFF);
        int e = be - 2; if (e<0) e=0; if (e>255) e=255;
        e8m0 = (unsigned char)e;
        int ne = 129 - be + 127; if (ne<1) ne=1; if (ne>254) ne=254;
        qs = __uint_as_float(((unsigned int)ne) << 23);
    }
    scale_out[0] = e8m0;

    // Quantize
    for (int i = 0; i < 16; i++) {
        unsigned char lo = sw_fp4(vals[2*i], qs);
        unsigned char hi = sw_fp4(vals[2*i+1], qs);
        fp4_out[i] = (lo & 0xF) | ((hi & 0xF) << 4);
    }
}

torch::Tensor run_dump(torch::Tensor A) {
    int K = A.size(1);
    auto fp4 = torch::zeros({16}, torch::dtype(torch::kUInt8).device(A.device()));
    auto sc = torch::zeros({1}, torch::dtype(torch::kUInt8).device(A.device()));
    hipLaunchKernelGGL(dump_quant, dim3(1), dim3(1), 0, 0,
        (const bf16*)A.data_ptr(), (unsigned char*)fp4.data_ptr(),
        (unsigned char*)sc.data_ptr(), K);
    return fp4;
}
"""

_CPP = r"""
#include <torch/extension.h>
torch::Tensor run_dump(torch::Tensor A);
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP, cuda_sources=_HIP_SRC,
                       functions=['run_dump'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '--offload-arch=gfx950'])
except Exception as e:
    print(f"[278] FAIL: {e}", file=sys.stderr)

_bc = [None, None, None]
_first = [True]

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
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _bc[0]:
        _bc[0] = dp
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)

    if _first[0] and _mod is not None and k <= 512:
        _first[0] = False
        # Dump HIP kernel's A FP4 bytes
        hip_fp4 = _mod.run_dump(A)
        torch.cuda.synchronize()
        hip_bytes = hip_fp4.cpu().tolist()

        # Reference: dynamic_mxfp4_quant
        A_q_ref, _ = dynamic_mxfp4_quant(A)
        ref_bytes = A_q_ref.view(torch.uint8)[0, :16].cpu().tolist()

        print(f"[278] HIP quant row0[:16]: {hip_bytes}", file=sys.stderr)
        print(f"[278] Ref quant row0[:16]: {ref_bytes}", file=sys.stderr)
        match = hip_bytes == ref_bytes
        diff_count = sum(1 for a, b in zip(hip_bytes, ref_bytes) if a != b)
        print(f"[278] Match: {match}, diff_count: {diff_count}/16", file=sys.stderr)

    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
