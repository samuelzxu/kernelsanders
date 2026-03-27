#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #225: Custom HIP MFMA FP4 GEMM - clean rewrite.
Minimal 32x32 single-wavefront kernel to test correctness first.
Falls back to preshuffle if HIP kernel fails.
"""
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

import json, sys
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

_CONFIGS = {
    "N=2880-K=512": {
        "M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=4096-K=512": {
        "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=7168-K=2048": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
    "N=3072-K=1536": {
        "M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"},
        "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}
    },
}

def _inject_configs():
    try:
        dev = arch_info.get_arch()
    except Exception:
        dev = "gfx950"
    config_dir = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
    os.makedirs(config_dir, exist_ok=True)
    for shape_key, config in _CONFIGS.items():
        fpath = f"{config_dir}/{dev}-GEMM-A16WFP4_PRESHUFFLED-{shape_key}.json"
        with open(fpath, "w") as f:
            json.dump(config, f)

try:
    _inject_configs()
except Exception:
    pass


# ── Minimal HIP FP4 GEMM: single wavefront 32x32 tile ──
# Completely rewritten to test correctness with simplest possible kernel.
# A: bf16, B_q: packed FP4, B_scale: E8M0 unshuffled
# Both A and B use same register layout: lane32=matrix row, group=K-half
# MFMA hardware transposes B to compute C = A @ B^T
_HIP_CODE = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <torch/extension.h>

/* MFMA FP4 GEMM kernel - build 225b - fixed mk_scale */
typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

// BF16 -> float
__device__ __forceinline__ float to_f32(bf16 v) { return static_cast<float>(v); }

// Float -> FP4 E2M1 (round to nearest)
__device__ __forceinline__ unsigned char to_fp4(float x) {
    unsigned char s = (x < 0.f) ? 8u : 0u;
    float a = fabsf(x);
    unsigned char c;
    if      (a < 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a < 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a < 2.5f)  c = 4;
    else if (a < 3.5f)  c = 5;
    else if (a < 5.0f)  c = 6;
    else                 c = 7;
    return s | c;
}

// Compute E8M0 scale: 2^(code-127) >= max_abs/6
__device__ __forceinline__ unsigned char mk_scale(float mx) {
    if (mx == 0.f) return 127u;
    int e = (int)ceilf(log2f(mx / 6.f)) + 127;
    if (e < 0) e = 0;
    if (e > 255) e = 255;
    return (unsigned char)e;
}

// E8M0 -> float
__device__ __forceinline__ float sc2f(unsigned char e) {
    return __uint_as_float(((unsigned int)e) << 23);
}

// Quantize 32 bf16 values to 16 packed FP4 bytes + return E8M0 scale
__device__ __forceinline__ unsigned char quant_group(
    const bf16* ptr, int valid, unsigned char* out16
) {
    float v[32]; float mx = 0.f;
    for (int i = 0; i < 32; i++) {
        v[i] = (i < valid) ? to_f32(ptr[i]) : 0.f;
        mx = fmaxf(mx, fabsf(v[i]));
    }
    unsigned char sc = mk_scale(mx);
    float inv = 1.f / sc2f(sc);
    for (int i = 0; i < 16; i++) {
        unsigned char lo = to_fp4(v[2*i] * inv);
        unsigned char hi = to_fp4(v[2*i+1] * inv);
        out16[i] = (lo & 0xFu) | ((hi & 0xFu) << 4);
    }
    return sc;
}

// Single-wavefront 32x32 GEMM tile
// A: (M,K) bf16, B: (N,K/2) uint8 packed FP4, Bs: (N,K/32) E8M0
// C: (M,N) bf16 output
// Grid: (N/32, M/32, ksplit)
extern "C" __global__ __launch_bounds__(64)
void mfma_fp4_gemm_32x32(
    const bf16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ Bs,
    bf16* __restrict__ C,
    float* __restrict__ Csplit,
    int M, int N, int K, int ksplit
) {
    int bn = blockIdx.x;
    int bm = blockIdx.y;
    int bk = blockIdx.z;
    int lane = threadIdx.x;  // 0..63
    int l32 = lane % 32;
    int grp = lane / 32;

    int m0 = bm * 32;
    int n0 = bn * 32;
    int Kh = K / 2;
    int Ks = K / 32;

    // K range
    int kps = ((Kh + ksplit - 1) / ksplit + 31) / 32 * 32;
    int ks = bk * kps;
    int ke = min(ks + kps, Kh);
    if (ks >= Kh) return;

    vf16 acc = {};
    for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks; kp < ke; kp += 32) {
        // -- A: quantize bf16 to FP4 --
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 8; i++) ab.v[i] = 0;
        unsigned char asc = 127u;
        {
            int r = m0 + l32;
            int koff = (kp + grp * 16) * 2;  // fp4 element offset
            if (r < M && koff < K) {
                int valid = min(32, K - koff);
                asc = quant_group(A + r * K + koff, valid, ab.b);
            }
        }

        // -- B: load packed FP4 directly (same layout as A) --
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 8; i++) bb.v[i] = 0;
        unsigned char bsc = 127u;
        {
            int nr = n0 + l32;
            int koff = kp + grp * 16;
            if (nr < N && koff < Kh) {
                int valid = min(16, Kh - koff);
                const unsigned char* bp = B + nr * Kh + koff;
                for (int i = 0; i < valid; i++) bb.b[i] = bp[i];
            }
            int ski = (kp + grp * 16) * 2 / 32;
            if (nr < N && ski < Ks)
                bsc = Bs[nr * Ks + ski];
        }

        // Exchange scales between group 0 and 1
        unsigned int oasc = __shfl_xor((unsigned int)asc, 32);
        unsigned int obsc = __shfl_xor((unsigned int)bsc, 32);
        unsigned char as0 = grp == 0 ? asc : (unsigned char)oasc;
        unsigned char as1 = grp == 0 ? (unsigned char)oasc : asc;
        unsigned char bs0 = grp == 0 ? bsc : (unsigned char)obsc;
        unsigned char bs1 = grp == 0 ? (unsigned char)obsc : bsc;

        // Pack 2 scales per i32: byte0=group0, byte1=group1
        int asp = (int)as0 | ((int)as1 << 8);
        int bsp = (int)bs0 | ((int)bs1 << 8);

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            ab.v, bb.v, acc, 4, 4, 0, asp, 0, bsp);
    }

    // Store: acc[i*4+j] -> row=(grp*4+i*8+j), col=l32
    if (ksplit > 1 && Csplit) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j;
                int c = n0 + l32;
                if (r < M && c < N)
                    atomicAdd(&Csplit[r*N+c], acc[i*4+j]);
            }
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j;
                int c = n0 + l32;
                if (r < M && c < N)
                    C[r*N+c] = bf16(acc[i*4+j]);
            }
    }
}

// Split-K reduce: f32 -> bf16
extern "C" __global__ void reduce_f32_bf16(
    const float* src, bf16* dst, int count
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < count) dst[i] = bf16(src[i]);
}

torch::Tensor hip_gemm(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int ks) {
    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    auto C = torch::empty({M, N}, A.options());
    dim3 g((N+31)/32, (M+31)/32, ks);
    if (ks > 1) {
        auto Cs = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
        hipLaunchKernelGGL(mfma_fp4_gemm_32x32, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
            (const unsigned char*)Bs.data_ptr(), (bf16*)C.data_ptr(),
            (float*)Cs.data_ptr(), M, N, K, ks);
        int tot = M*N;
        hipLaunchKernelGGL(reduce_f32_bf16, dim3((tot+255)/256), dim3(256), 0, 0,
            (const float*)Cs.data_ptr(), (bf16*)C.data_ptr(), tot);
    } else {
        hipLaunchKernelGGL(mfma_fp4_gemm_32x32, g, dim3(64), 0, 0,
            (const bf16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
            (const unsigned char*)Bs.data_ptr(), (bf16*)C.data_ptr(),
            (float*)nullptr, M, N, K, 1);
    }
    return C;
}
"""

_CPP_CODE = r"""
#include <torch/extension.h>
torch::Tensor hip_gemm(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int ks);
"""

_mod = None
try:
    _mod = load_inline(
        name='mfma_fp4_225b',
        cpp_sources=_CPP_CODE,
        cuda_sources=_HIP_CODE,
        functions=['hip_gemm'],
        verbose=True,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    print("[225] HIP MFMA kernel compiled OK", file=sys.stderr)
except Exception as e:
    print(f"[225] HIP compile FAIL: {e}", file=sys.stderr)


# Preshuffle fallback
_bc = [None, None, None]

def _ps_gemm(A, Bsh, Bssh, m, k, n):
    key = (Bsh.data_ptr(), Bssh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = Bsh.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = Bssh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)


def _unshuffle_scale(ssh, n, k):
    s = ssh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm//32, sn//8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    s = s.view(sm, sn)
    return s[:n, :k//32].contiguous()


_KSPLIT = {
    (4, 2880, 512): 1, (16, 2112, 7168): 4,
    (32, 4096, 512): 1, (32, 2880, 512): 1,
    (64, 7168, 2048): 2, (256, 3072, 1536): 3,
}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _mod is not None:
        ks = _KSPLIT.get((m, n, k), 1)
        bq = B_q.view(torch.uint8)
        bs = _unshuffle_scale(B_scale_sh, n, k)
        return _mod.hip_gemm(A, bq, bs, ks)

    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
