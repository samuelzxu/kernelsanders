#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #248: Use aiter's dynamic_mxfp4_quant for SCALE ONLY, then hw intrinsic for FP4 conversion.
This isolates the rounding issue: if scales match perfectly but FP4 values differ,
the hw intrinsic rounds differently from the reference.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant

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

_MODULE = 'mfma248b'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;
typedef __attribute__((ext_vector_type(2))) __bf16 nbf16x2;

__device__ __forceinline__ float e8m0_to_f(unsigned char e) {
    return __uint_as_float(((unsigned int)e) << 23);
}

// Quantize 32 bf16 -> 16 packed FP4 bytes using hw intrinsic + EXTERNAL scale
__device__ void hw_quant_with_scale(const bf16* src, int valid, float scale, unsigned char* out16) {
    unsigned int u[4] = {0, 0, 0, 0};
    #define CVT(bi) do { \
        __bf16 _v0 = (2*(bi) < valid) ? (__bf16)static_cast<float>(src[2*(bi)]) : (__bf16)0.0f; \
        __bf16 _v1 = (2*(bi)+1 < valid) ? (__bf16)static_cast<float>(src[2*(bi)+1]) : (__bf16)0.0f; \
        nbf16x2 _p = {_v0, _v1}; \
        u[(bi)/4] = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u[(bi)/4], _p, scale, (bi)%4); \
    } while(0)
    CVT(0); CVT(1); CVT(2); CVT(3); CVT(4); CVT(5); CVT(6); CVT(7);
    CVT(8); CVT(9); CVT(10); CVT(11); CVT(12); CVT(13); CVT(14); CVT(15);
    #undef CVT
    unsigned char* up = (unsigned char*)u;
    for (int i = 0; i < 16; i++) out16[i] = up[i];
}

// Kernel: takes bf16 A + EXTERNAL A_scale (from dynamic_mxfp4_quant) + pre-quantized B
extern "C" __global__ __launch_bounds__(64)
void kern248(
    const bf16* __restrict__ A,
    const unsigned char* __restrict__ As,   // (M, K/32) E8M0 from dynamic_mxfp4_quant
    const unsigned char* __restrict__ Bq,   // (N, K/2) packed FP4
    const unsigned char* __restrict__ Bs,   // (N, K/32) E8M0
    bf16* __restrict__ C, float* __restrict__ Cs,
    int M, int N, int K, int ksplit
) {
    int bn = blockIdx.x, bm = blockIdx.y, bk = blockIdx.z;
    int lane = threadIdx.x, l32 = lane % 32, grp = lane / 32;
    int m0 = bm * 32, n0 = bn * 32;
    int Kh = K / 2, Ks = K / 32;

    int kps = ((Kh + ksplit - 1) / ksplit + 31) / 32 * 32;
    int ks_start = bk * kps, ks_end = ks_start + kps;
    if (ks_end > Kh) ks_end = Kh;
    if (ks_start >= Kh) return;

    vf16 acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks_start; kp < ks_end; kp += 32) {
        // A: read EXTERNAL scale, then hw-quant bf16 to FP4
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 8; i++) ab.v[i] = 0;
        unsigned char asc = 127u;
        {
            int r = m0 + l32;
            int k_fp4 = (kp + grp * 16) * 2;
            int ski = (kp + grp * 16) * 2 / 32;
            // Read scale from external source (dynamic_mxfp4_quant output)
            if (r < M && ski < Ks)
                asc = As[(long)r * Ks + ski];
            float scale = e8m0_to_f(asc);
            if (r < M && k_fp4 < K) {
                int valid = K - k_fp4; if (valid > 32) valid = 32;
                hw_quant_with_scale(A + (long)r * K + k_fp4, valid, scale, ab.b);
            }
        }

        // B: load pre-quantized FP4
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 8; i++) bb.v[i] = 0;
        unsigned char bsc = 127u;
        {
            int nr = n0 + l32, koff = kp + grp * 16;
            if (nr < N && koff < Kh) {
                const unsigned char* bp = Bq + (long)nr * Kh + koff;
                int vld = Kh - koff; if (vld > 16) vld = 16;
                for (int i = 0; i < vld; i++) bb.b[i] = bp[i];
            }
            int ski = (kp + grp * 16) * 2 / 32;
            if (n0 + l32 < N && ski < Ks)
                bsc = Bs[(long)(n0 + l32) * Ks + ski];
        }

        unsigned int oasc = __shfl_xor((unsigned int)asc, 32);
        unsigned int obsc = __shfl_xor((unsigned int)bsc, 32);
        unsigned char as0 = grp == 0 ? asc : (unsigned char)oasc;
        unsigned char as1 = grp == 0 ? (unsigned char)oasc : asc;
        unsigned char bs0 = grp == 0 ? bsc : (unsigned char)obsc;
        unsigned char bs1 = grp == 0 ? (unsigned char)obsc : bsc;
        int asp = (int)as0 | ((int)as1 << 8);
        int bsp = (int)bs0 | ((int)bs1 << 8);

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            ab.v, bb.v, acc, 4, 4, 0, asp, 0, bsp);
    }

    if (ksplit > 1 && Cs) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j, c = n0 + l32;
                if (r < M && c < N) atomicAdd(&Cs[(long)r*N+c], acc[i*4+j]);
            }
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j, c = n0 + l32;
                if (r < M && c < N) C[(long)r*N+c] = bf16(acc[i*4+j]);
            }
    }
}

extern "C" __global__ void kern248_red(const float* s, bf16* d, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = bf16(s[i]);
}

torch::Tensor run248(torch::Tensor A, torch::Tensor As,
                     torch::Tensor Bq, torch::Tensor Bs, int ks) {
    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
        hipLaunchKernelGGL(kern248, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bq.data_ptr(), (const unsigned char*)Bs.data_ptr(),
            (bf16*)C.data_ptr(), (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M * N;
        hipLaunchKernelGGL(kern248_red, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(kern248, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)As.data_ptr(),
            (const unsigned char*)Bq.data_ptr(), (const unsigned char*)Bs.data_ptr(),
            (bf16*)C.data_ptr(), (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run248(torch::Tensor A, torch::Tensor As, torch::Tensor Bq, torch::Tensor Bs, int ks);
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['run248'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
    print("[248] OK", file=sys.stderr)
except Exception as e:
    print(f"[248] FAIL: {e}", file=sys.stderr)

_bc = [None, None, None]
def _ps_gemm(A, Bsh, Bssh, m, k, n):
    key = (Bsh.data_ptr(), Bssh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = Bsh.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = Bssh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)

def _unshuffle(ssh, n, k):
    s = ssh.view(torch.uint8)
    sm, sn = s.shape
    return s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:n, :k//32].contiguous()

_KS = {(4,2880,512):1,(16,2112,7168):4,(32,4096,512):1,(32,2880,512):1,(64,7168,2048):2,(256,3072,1536):3}

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    if _mod is not None:
        # Get A scale from dynamic_mxfp4_quant (matches reference exactly)
        _, A_scale = dynamic_mxfp4_quant(A)
        a_sc = A_scale.view(torch.uint8)[:m, :k//32].contiguous()

        # Get FRESH B quant (bypass B_q/B_scale_sh from task - use dynamic_mxfp4_quant directly)
        B_q_fresh, B_scale_fresh = dynamic_mxfp4_quant(B)
        bq = B_q_fresh.view(torch.uint8)
        bs = B_scale_fresh.view(torch.uint8)[:n, :k//32].contiguous()
        ks = _KS.get((m, n, k), 1)
        return _mod.run248(A, a_sc, bq, bs, ks)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
