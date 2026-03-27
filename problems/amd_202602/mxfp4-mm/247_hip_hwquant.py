#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #247: HIP MFMA with HARDWARE FP4 quantization intrinsic.

Key fix: use __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16 for bf16->FP4 conversion.
This is the SAME hardware instruction that aiter/CK uses. Previous attempts used
software rounding which differs from hardware rounding.

Combined with confirmed-correct MFMA register layout from probes #240/#241.
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

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

_MODULE = 'mfma247i'
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

/* build 247: hardware FP4 quantization + confirmed MFMA layout */
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;
typedef __attribute__((ext_vector_type(2))) __bf16 native_bf16x2;

// E8M0 scale computation: scale = 2^(e-127) where e = ceil(log2(max_abs/6)) + 127
// OCP MXFP4 E8M0 scale: shared_exp = floor(log2(max_abs)), e8m0 = shared_exp + 127
// The hw cvt intrinsic divides each value by scale = 2^shared_exp = 2^(e8m0-127)
// and rounds to nearest FP4 E2M1 value [0, 0.5, 1, 1.5, 2, 3, 4, 6]
// Since max_abs / 2^floor(log2(max_abs)) is in [1, 2), the scaled value is in [1, 2)
// which is well within FP4 range (max 6.0). This is the standard OCP formula.
// EXACT formula from aiter _mxfp4_quant_op (extracted from runner source):
// Round amax up, then biased_exp - 2
__device__ __forceinline__ unsigned char mk_e8m0(float max_abs) {
    if (max_abs == 0.f) return 0u;
    unsigned int ai = __float_as_uint(max_abs);
    ai = (ai + 0x200000u) & 0xFF800000u;  // round up to power-of-2
    int e8m0 = (int)((ai >> 23) & 0xFF) - 2;
    if (e8m0 < 0) e8m0 = 0;
    if (e8m0 > 255) e8m0 = 255;
    return (unsigned char)e8m0;
}

__device__ __forceinline__ float e8m0_to_f(unsigned char e) {
    return __uint_as_float(((unsigned int)e) << 23);
}

// Quantize 32 bf16 values to 16 packed FP4 bytes using HARDWARE intrinsic
// __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u32, bf16x2, scale, word_idx)
// Converts 2 bf16 values to 2 FP4 nibbles, packed into specified byte of u32
// scale is the INVERSE scale (1/2^(e8m0-127)) applied during conversion
__device__ void hw_quant_32(const bf16* src, int valid, unsigned char* out16, unsigned char* out_scale) {
    // Step 1: find max abs of 32 values
    float max_abs = 0.f;
    for (int i = 0; i < 32 && i < valid; i++) {
        float v = static_cast<float>(src[i]);
        float av = fabsf(v);
        if (av > max_abs) max_abs = av;
    }

    // Step 2: compute E8M0 scale
    unsigned char e8m0 = mk_e8m0(max_abs);
    *out_scale = e8m0;
    // The cvt intrinsic divides by scale: fp4 = round(bf16 / scale)
    float scale = e8m0_to_f(e8m0);  // 2^(e8m0-127)

    // Step 3: hardware conversion - packs 2 FP4 per call into a u32
    // The intrinsic: __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u32, bf16x2, scale, word_idx)
    // word_idx selects which byte of u32 to write the 2 nibbles into
    // Each call processes 2 bf16 values -> 2 FP4 nibbles -> 1 byte
    // We need 16 bytes = 16 calls, packing into 4 u32s
    // Must unroll with constant word_idx (last arg must be compile-time constant)
    unsigned int u[4] = {0, 0, 0, 0};
    #define CVT_PAIR(byte_idx) do { \
        __bf16 _v0 = (2*(byte_idx) < valid) ? (__bf16)static_cast<float>(src[2*(byte_idx)]) : (__bf16)0.0f; \
        __bf16 _v1 = (2*(byte_idx)+1 < valid) ? (__bf16)static_cast<float>(src[2*(byte_idx)+1]) : (__bf16)0.0f; \
        native_bf16x2 _p = {_v0, _v1}; \
        u[(byte_idx)/4] = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u[(byte_idx)/4], _p, scale, (byte_idx)%4); \
    } while(0)
    CVT_PAIR(0);  CVT_PAIR(1);  CVT_PAIR(2);  CVT_PAIR(3);
    CVT_PAIR(4);  CVT_PAIR(5);  CVT_PAIR(6);  CVT_PAIR(7);
    CVT_PAIR(8);  CVT_PAIR(9);  CVT_PAIR(10); CVT_PAIR(11);
    CVT_PAIR(12); CVT_PAIR(13); CVT_PAIR(14); CVT_PAIR(15);
    #undef CVT_PAIR

    // Copy u32s to output bytes
    unsigned char* up = (unsigned char*)u;
    for (int i = 0; i < 16; i++) {
        out16[i] = up[i];
    }
}

// 32x32 tile, 1 wavefront
extern "C" __global__ __launch_bounds__(64)
void kern247(
    const bf16* __restrict__ A,          // (M, K) bf16
    const unsigned char* __restrict__ Bq, // (N, K/2) packed FP4
    const unsigned char* __restrict__ Bs, // (N, K/32) E8M0
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
        // A: inline bf16 -> FP4 quant using HARDWARE intrinsic
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 8; i++) ab.v[i] = 0;
        unsigned char asc = 127u;
        {
            int r = m0 + l32;
            int k_fp4 = (kp + grp * 16) * 2;  // FP4 element index
            if (r < M && k_fp4 < K) {
                int valid = K - k_fp4; if (valid > 32) valid = 32;
                hw_quant_32(A + (long)r * K + k_fp4, valid, ab.b, &asc);
            }
        }

        // B: load pre-quantized FP4 directly
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

        // Exchange scales
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

extern "C" __global__ void kern247_red(const float* s, bf16* d, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) d[i] = bf16(s[i]);
}

torch::Tensor run247(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int ks) {
    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
        hipLaunchKernelGGL(kern247, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
            (const unsigned char*)Bs.data_ptr(), (bf16*)C.data_ptr(),
            (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M * N;
        hipLaunchKernelGGL(kern247_red, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(kern247, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
            (const unsigned char*)Bs.data_ptr(), (bf16*)C.data_ptr(),
            (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run247(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int ks);
"""

_cb = os.path.expanduser("~/.cache/torch_extensions")
for d in os.listdir(_cb) if os.path.isdir(_cb) else []:
    cd = os.path.join(_cb, d, _MODULE)
    if os.path.isdir(cd): shutil.rmtree(cd, ignore_errors=True)

_mod = None
try:
    _mod = load_inline(name=_MODULE, cpp_sources=_CPP_SRC, cuda_sources=_HIP_SRC,
                       functions=['run247'], verbose=True,
                       extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'])
    print("[247] OK", file=sys.stderr)
except Exception as e:
    print(f"[247] FAIL: {e}", file=sys.stderr)

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
        bq = B_q.view(torch.uint8)
        bs = _unshuffle(B_scale_sh, n, k)
        ks = _KS.get((m, n, k), 1)
        return _mod.run247(A, bq, bs, ks)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
