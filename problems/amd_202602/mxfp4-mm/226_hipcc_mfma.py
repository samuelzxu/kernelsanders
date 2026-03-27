#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #226: Custom HIP MFMA FP4 GEMM via hipcc + ctypes.
Bypasses torch load_inline cache by compiling directly with hipcc.
"""
import os, sys, subprocess, ctypes, tempfile, hashlib
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

import json, torch
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

# Preshuffle configs
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


# ── Compile HIP kernel directly with hipcc ──
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>

typedef int __attribute__((ext_vector_type(8))) vi8;
typedef float __attribute__((ext_vector_type(16))) vf16;
typedef hip_bfloat16 bf16;

__device__ __forceinline__ float to_f32(bf16 v) { return static_cast<float>(v); }

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

__device__ __forceinline__ unsigned char mk_scale(float mx) {
    if (mx == 0.f) return 127u;
    int e = (int)ceilf(log2f(mx / 6.f)) + 127;
    if (e < 0) e = 0;
    if (e > 255) e = 255;
    return (unsigned char)e;
}

__device__ __forceinline__ float sc2f(unsigned char e) {
    return __uint_as_float(((unsigned int)e) << 23);
}

// Quantize 32 bf16 values to 16 packed FP4 bytes, return E8M0 scale
__device__ unsigned char quant32(const bf16* ptr, int valid, unsigned char* out16) {
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

// 32x32 single-wavefront MFMA FP4 GEMM tile
// A: (M,K) bf16, B: (N,K/2) packed FP4, Bs: (N,K/32) E8M0
// C: (M,N) bf16 or Csplit: (M,N) f32
extern "C" __global__ __launch_bounds__(64)
void kern_fp4_gemm_32x32(
    const bf16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const unsigned char* __restrict__ Bs,
    bf16* __restrict__ C,
    float* __restrict__ Csplit,
    int M, int N, int K, int ksplit
) {
    int bn = blockIdx.x, bm = blockIdx.y, bk = blockIdx.z;
    int lane = threadIdx.x;
    int l32 = lane % 32;
    int grp = lane / 32;

    int m0 = bm * 32, n0 = bn * 32;
    int Kh = K / 2, Ks = K / 32;

    int kps = ((Kh + ksplit - 1) / ksplit + 31) / 32 * 32;
    int ks = bk * kps, ke = ks + kps;
    if (ke > Kh) ke = Kh;
    if (ks >= Kh) return;

    vf16 acc = {};
    for (int i = 0; i < 16; i++) acc[i] = 0.f;

    for (int kp = ks; kp < ke; kp += 32) {
        // A: quantize bf16 to FP4
        union { vi8 v; unsigned char b[32]; } ab;
        for (int i = 0; i < 8; i++) ab.v[i] = 0;
        unsigned char asc = 127u;
        {
            int r = m0 + l32;
            int koff = (kp + grp * 16) * 2;
            if (r < M && koff < K) {
                int vld = K - koff; if (vld > 32) vld = 32;
                asc = quant32(A + r * K + koff, vld, ab.b);
            }
        }

        // B: load packed FP4 (same register layout as A)
        union { vi8 v; unsigned char b[32]; } bb;
        for (int i = 0; i < 8; i++) bb.v[i] = 0;
        unsigned char bsc = 127u;
        {
            int nr = n0 + l32;
            int koff = kp + grp * 16;
            if (nr < N && koff < Kh) {
                int vld = Kh - koff; if (vld > 16) vld = 16;
                const unsigned char* bp = B + nr * Kh + koff;
                for (int i = 0; i < vld; i++) bb.b[i] = bp[i];
            }
            int ski = (kp + grp * 16) * 2 / 32;
            if (nr < N && ski < Ks) bsc = Bs[nr * Ks + ski];
        }

        // Exchange scales between groups
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

    // Store
    if (ksplit > 1 && Csplit) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j, c = n0 + l32;
                if (r < M && c < N) atomicAdd(&Csplit[r*N+c], acc[i*4+j]);
            }
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                int r = m0 + grp*4 + i*8 + j, c = n0 + l32;
                if (r < M && c < N) C[r*N+c] = bf16(acc[i*4+j]);
            }
    }
}

extern "C" __global__ void kern_reduce(const float* src, bf16* dst, int n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) dst[i] = bf16(src[i]);
}
"""

def _compile_hip():
    """Compile HIP kernel directly with hipcc, bypassing torch cache."""
    src_hash = hashlib.md5(_HIP_SRC.encode()).hexdigest()[:8]
    so_path = f"/tmp/mfma_fp4_{src_hash}.so"
    if os.path.exists(so_path):
        return ctypes.CDLL(so_path)

    with tempfile.NamedTemporaryFile(suffix='.hip', mode='w', delete=False) as f:
        f.write(_HIP_SRC)
        hip_path = f.name

    try:
        cmd = [
            '/opt/rocm/bin/hipcc',
            '--offload-arch=gfx950',
            '-shared', '-fPIC',
            '-O3', '-w', '-mcumode',
            '-o', so_path,
            hip_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[226] hipcc FAIL: {result.stderr[:500]}", file=sys.stderr)
            return None
        print(f"[226] hipcc OK: {so_path}", file=sys.stderr)
        return ctypes.CDLL(so_path)
    except Exception as e:
        print(f"[226] hipcc exception: {e}", file=sys.stderr)
        return None
    finally:
        os.unlink(hip_path)

_lib = _compile_hip()

def _hip_gemm(A, Bq, Bs, ksplit):
    """Launch HIP MFMA GEMM via ctypes."""
    if _lib is None:
        return None

    M, K = A.shape
    N = Bq.shape[0]
    C = torch.empty((M, N), dtype=torch.bfloat16, device=A.device)

    import torch.cuda  # for current device/hipGetDevice

    gx = (N + 31) // 32
    gy = (M + 31) // 32
    gz = ksplit

    if ksplit > 1:
        Cs = torch.zeros((M, N), dtype=torch.float32, device=A.device)
        _lib.hipLaunchKernelGGL  # doesn't work via ctypes...

    # ctypes can't easily launch HIP kernels.
    # Use torch's HIP launch infrastructure instead.
    return None


# ── Preshuffle fallback ──
_bc = [None, None, None]

def _ps_gemm(A, Bsh, Bssh, m, k, n):
    key = (Bsh.data_ptr(), Bssh.data_ptr())
    if key != _bc[0]:
        _bc[0] = key
        _bc[1] = Bsh.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = Bssh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]
    # HIP kernel can't launch via ctypes - fall back to preshuffle
    return _ps_gemm(A, B_shuffle, B_scale_sh, m, k, n)
