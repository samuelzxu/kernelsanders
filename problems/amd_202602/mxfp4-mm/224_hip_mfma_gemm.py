#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4-MM #224: Custom HIP MFMA FP4 GEMM kernel via load_inline.

Strategy: Use custom HIP kernel for ALL shapes, with inline BF16->FP4 quant.
Falls back to preshuffle if HIP kernel fails to compile.

Key insight: __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 processes 64 FP4
elements (2 scale groups of 32). Pack both scales into the int parameter:
  scale_packed = scale_group0 | (scale_group1 << 8)
with opsel=0 to use bytes [0,1].
"""
import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ['HSA_XNACK'] = '0'

import json, sys
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# ── Inject preshuffle configs (fallback path) ──
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

# ── Custom HIP MFMA FP4 GEMM Kernel ──
_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hip/amd_detail/amd_hip_bfloat16.h>

typedef int __attribute__((ext_vector_type(8))) i32x8_t;
typedef float __attribute__((ext_vector_type(16))) f32x16_t;
typedef hip_bfloat16 bf16_t;

#define FP4_TYPE 4  // E2M1
#define KERNEL_VERSION 4  // force recompile

// ── BF16->FP4 E2M1 quantization ──

__device__ __forceinline__ float bf16_to_f32(bf16_t v) {
    return static_cast<float>(v);
}

// Round-to-nearest FP4 E2M1 quantization
// FP4 E2M1 positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
__device__ __forceinline__ unsigned char f32_to_fp4(float x) {
    unsigned char sign = (x < 0.0f) ? 8u : 0u;
    float ax = fabsf(x);
    unsigned char code;
    if      (ax < 0.25f)  code = 0;  // -> 0
    else if (ax < 0.75f)  code = 1;  // -> 0.5
    else if (ax < 1.25f)  code = 2;  // -> 1.0
    else if (ax < 1.75f)  code = 3;  // -> 1.5
    else if (ax < 2.5f)   code = 4;  // -> 2.0
    else if (ax < 3.5f)   code = 5;  // -> 3.0
    else if (ax < 5.0f)   code = 6;  // -> 4.0
    else                   code = 7;  // -> 6.0
    return sign | code;
}

__device__ __forceinline__ unsigned char compute_e8m0_scale(float max_abs) {
    if (max_abs == 0.0f) return 127u;
    // Want: max_abs / 2^(exp-127) <= 6.0
    // exp >= ceil(log2(max_abs / 6.0)) + 127
    int exp_val = (int)ceilf(log2f(max_abs * 0.16666667f)) + 127;
    if (exp_val < 0) exp_val = 0;
    if (exp_val > 255) exp_val = 255;
    return (unsigned char)exp_val;
}

__device__ __forceinline__ float e8m0_to_f32(unsigned char e) {
    unsigned int bits = ((unsigned int)e) << 23;
    return __uint_as_float(bits);
}

// ── Main kernel: 32x128 tile, 4 wavefronts ──
// MFMA computes C[32x32] = A[32xK] @ B[32xK]^T
// Both A and B use SAME register layout: lane32=row, group=K-half
// The hardware transposes B internally
extern "C"
__global__ __launch_bounds__(256, 2)
void fp4_gemm_a16wfp4_32x128(
    const bf16_t* __restrict__ A,     // (M, K) bf16
    const unsigned char* __restrict__ B_q,    // (N, K/2) packed FP4
    const unsigned char* __restrict__ B_scale, // (N, K/32) E8M0
    bf16_t* __restrict__ C,           // (M, N) bf16
    float* __restrict__ C_split,              // (M, N) f32 for split-K, or nullptr
    const int M, const int N, const int K,
    const int num_ksplit
) {
    const int bid_n = blockIdx.x;
    const int bid_m = blockIdx.y;
    const int bid_k = blockIdx.z;

    const int tid = threadIdx.x;       // [0, 255]
    const int warp_id = tid / 64;      // [0, 3]
    const int lane = tid % 64;
    const int lane32 = lane % 32;
    const int group = lane / 32;       // 0 or 1

    const int TILE_M = 32;
    const int TILE_N = 128;

    const int m_start = bid_m * TILE_M;
    const int n_start = bid_n * TILE_N + warp_id * 32;  // each warp handles 32 N cols

    // K range for this split
    const int K_half = K / 2;
    const int k_per_split = ((K_half + num_ksplit - 1) / num_ksplit);
    // Align to 32 bytes (64 FP4 = 1 MFMA K-block)
    const int k_aligned = ((k_per_split + 31) / 32) * 32;
    const int k_start = bid_k * k_aligned;
    const int k_end_raw = k_start + k_aligned;
    const int k_end = k_end_raw < K_half ? k_end_raw : K_half;

    if (k_start >= K_half) return;

    f32x16_t acc = {};
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    const int K_scale = K / 32;

    // ── K loop ──
    for (int kp = k_start; kp < k_end; kp += 32) {
        // kp = packed byte offset in K dimension
        // Each iteration: 32 packed bytes = 64 FP4 values = 2 scale groups

        // ── Load A: quantize bf16 -> FP4 inline ──
        // lane32 selects row [0..31], group selects 16-byte half [0 or 1]
        union { i32x8_t v; unsigned char b[32]; } a_buf;
        unsigned char a_sc;
        #pragma unroll
        for (int i = 0; i < 8; i++) a_buf.v[i] = 0;
        a_sc = 127u;

        {
            const int row = m_start + lane32;
            const int k_byte_off = kp + group * 16;  // byte offset in packed K
            const int k_fp4 = k_byte_off * 2;        // FP4 element index

            if (row < M && k_byte_off < K_half) {
                const bf16_t* a_ptr = A + row * K + k_fp4;

                float vals[32];
                float max_abs = 0.0f;
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    vals[i] = (k_fp4 + i < K) ? bf16_to_f32(a_ptr[i]) : 0.0f;
                    float av = fabsf(vals[i]);
                    if (av > max_abs) max_abs = av;
                }

                a_sc = compute_e8m0_scale(max_abs);
                float inv_sc = 1.0f / e8m0_to_f32(a_sc);

                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    unsigned char lo = f32_to_fp4(vals[i*2] * inv_sc);
                    unsigned char hi = f32_to_fp4(vals[i*2+1] * inv_sc);
                    a_buf.b[i] = (lo & 0xFu) | ((hi & 0xFu) << 4);
                }
            }
        }

        // Exchange A scales between group 0 and group 1 via warp shuffle
        unsigned char a_scale0, a_scale1;
        {
            unsigned int other_sc = __shfl_xor((unsigned int)a_sc, 32);
            a_scale0 = (group == 0) ? a_sc : (unsigned char)other_sc;
            a_scale1 = (group == 0) ? (unsigned char)other_sc : a_sc;
        }

        // ── Load B: same layout as A (lane32=N-row, group=K-half) ──
        // MFMA B uses identical register layout to A
        // The hardware transposes B to compute C = A @ B^T
        union { i32x8_t v; unsigned char b[32]; } b_buf;
        unsigned char b_sc;
        #pragma unroll
        for (int i = 0; i < 8; i++) b_buf.v[i] = 0;
        b_sc = 127u;

        {
            const int b_n = n_start + lane32;  // lane32 selects N-row
            const int k_byte_off = kp + group * 16;

            if (b_n < N && k_byte_off < K_half) {
                const unsigned char* b_ptr = B_q + b_n * K_half + k_byte_off;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    b_buf.b[i] = b_ptr[i];
                }
            }

            // B scale
            const int scale_k0 = (kp + group * 16) * 2 / 32;  // FP4 index / 32
            if (b_n < N && scale_k0 < K_scale) {
                b_sc = B_scale[b_n * K_scale + scale_k0];
            }
        }

        // Exchange B scales between groups
        unsigned char b_scale0, b_scale1;
        {
            unsigned int other_sc = __shfl_xor((unsigned int)b_sc, 32);
            b_scale0 = (group == 0) ? b_sc : (unsigned char)other_sc;
            b_scale1 = (group == 0) ? (unsigned char)other_sc : b_sc;
        }

        // ── Pack scales: 2 E8M0 values per MFMA call ──
        // Byte 0 = scale for K[0..31], Byte 1 = scale for K[32..63]
        // Each lane provides its own scale (per-row for A, per-col for B)
        int a_scale_packed = (int)a_scale0 | ((int)a_scale1 << 8);
        int b_scale_packed = (int)b_scale0 | ((int)b_scale1 << 8);
        // DEBUG: try uniform scale (average of both) to test basic correctness
        // int a_scale_packed = (int)a_scale0;
        // int b_scale_packed = (int)b_scale0;

        // ── MFMA ──
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_buf.v, b_buf.v, acc,
            FP4_TYPE, FP4_TYPE,
            0, a_scale_packed,
            0, b_scale_packed
        );

        // No LDS sync needed - all loads are direct from global memory
    }

    // ── Store results ──
    // MFMA 32x32 output mapping:
    // acc[i*4+j] -> row = (group*4 + i*8 + j), col = lane32
    if (num_ksplit > 1 && C_split != nullptr) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int out_row = m_start + group * 4 + i * 8 + j;
                int out_col = n_start + lane32;
                if (out_row < M && out_col < N) {
                    atomicAdd(&C_split[out_row * N + out_col], acc[i * 4 + j]);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int out_row = m_start + group * 4 + i * 8 + j;
                int out_col = n_start + lane32;
                if (out_row < M && out_col < N) {
                    C[out_row * N + out_col] = bf16_t(acc[i * 4 + j]);
                }
            }
        }
    }
}

// ── Split-K reduction: f32 -> bf16 ──
extern "C"
__global__ void splitk_reduce_bf16(
    const float* __restrict__ src,
    bf16_t* __restrict__ dst,
    const int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = bf16_t(src[idx]);
    }
}

// ── C++ wrapper ──
torch::Tensor run_hip_gemm(
    torch::Tensor A,        // (M, K) bf16
    torch::Tensor B_q,      // (N, K/2) uint8
    torch::Tensor B_scale,  // (N, K/32) uint8
    int num_ksplit
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B_q.size(0);

    auto C = torch::empty({M, N}, A.options());  // bf16

    const int TILE_M = 32;
    const int TILE_N = 128;

    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M,
        num_ksplit
    );
    dim3 block(256);

    if (num_ksplit > 1) {
        auto C_split = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
        hipLaunchKernelGGL(fp4_gemm_a16wfp4_32x128, grid, block, 0, 0,
            (const bf16_t*)A.data_ptr(),
            (const unsigned char*)B_q.data_ptr(),
            (const unsigned char*)B_scale.data_ptr(),
            (bf16_t*)C.data_ptr(),
            (float*)C_split.data_ptr(),
            M, N, K, num_ksplit);

        int total = M * N;
        hipLaunchKernelGGL(splitk_reduce_bf16, dim3((total+255)/256), dim3(256), 0, 0,
            (const float*)C_split.data_ptr(),
            (bf16_t*)C.data_ptr(),
            total);
    } else {
        hipLaunchKernelGGL(fp4_gemm_a16wfp4_32x128, grid, block, 0, 0,
            (const bf16_t*)A.data_ptr(),
            (const unsigned char*)B_q.data_ptr(),
            (const unsigned char*)B_scale.data_ptr(),
            (bf16_t*)C.data_ptr(),
            (float*)nullptr,
            M, N, K, 1);
    }

    return C;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor run_hip_gemm(torch::Tensor A, torch::Tensor B_q, torch::Tensor B_scale, int num_ksplit);
"""

_hip_mod = None
try:
    _hip_mod = load_inline(
        name='fp4_hip_gemm_v4',
        cpp_sources=_CPP_SRC,
        cuda_sources=_HIP_SRC,
        functions=['run_hip_gemm'],
        verbose=True,
        extra_cuda_cflags=['-O3', '-w', '-mcumode', '--offload-arch=gfx950'],
    )
    print("[HIP] Custom FP4 GEMM v2 compiled!", file=sys.stderr)
except Exception as e:
    print(f"[HIP] Compilation failed: {e}", file=sys.stderr)


# ── Preshuffle fallback ──
_b_cache_key = None
_b_cache_w = None
_b_cache_ws = None

def _preshuffle_gemm(A, B_shuffle, B_scale_sh, m, k, n):
    global _b_cache_key, _b_cache_w, _b_cache_ws
    key = (B_shuffle.data_ptr(), B_scale_sh.data_ptr())
    if key == _b_cache_key:
        B_w, B_ws = _b_cache_w, _b_cache_ws
    else:
        B_w = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        B_ws = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)
        _b_cache_key, _b_cache_w, _b_cache_ws = key, B_w, B_ws
    return gemm_a16wfp4_preshuffle(A, B_w, B_ws, prequant=True, dtype=torch.bfloat16)

def _e8m0_unshuffle(scale_sh, n, k):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    s = s.view(sm, sn)
    return s[:n, :k // 32].contiguous()


# ── Shape-specific KSPLIT config ──
_KSPLIT = {
    (4, 2880, 512): 1,
    (16, 2112, 7168): 4,
    (32, 4096, 512): 1,
    (32, 2880, 512): 1,
    (64, 7168, 2048): 2,
    (256, 3072, 1536): 3,
}

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Try custom HIP kernel
    if _hip_mod is not None:
        ksplit = _KSPLIT.get((m, n, k), 1)
        B_q_uint8 = B_q.view(torch.uint8)
        B_scale = _e8m0_unshuffle(B_scale_sh, n, k)
        return _hip_mod.run_hip_gemm(A, B_q_uint8, B_scale, ksplit)

    # Fallback to preshuffle
    return _preshuffle_gemm(A, B_shuffle, B_scale_sh, m, k, n)
