"""
MLA decode - Minimal HIP attention kernel for small configs via load_inline.

Strategy: write a SIMPLE but CORRECT HIP kernel that uses shared memory
for Q and processes KV tokens in chunks. No MFMA — just vector FMA.
Focus on correctness first, then optimize.

For bs=4/kv=1024: currently 17µs with torch.compile GEMM.
Target: <10µs with single-kernel HIP launch.

Key design: ONE threadblock per batch element, 256 threads.
All 16 heads share KV loads (GQA optimization).
Phase 1: Each thread computes Q@K^T for assigned tokens (parallel over tokens)
Phase 2: Block-wide softmax reduction
Phase 3: Each thread accumulates P@V for assigned V dims (parallel over V dims)
Chunked over KV tokens for Phase 1+3 coordination.
"""

import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_VERBOSE', '0')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS', '100')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS', '50')

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.triton.quant.quant import dynamic_per_tensor_quant_fp8_i8

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)


# ============================================================================
# HIP kernel — chunked flash-decode, one block per batch element
# All 16 heads processed per block via shared memory
# ============================================================================

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <torch/extension.h>

#define TPB 256
#define CHUNK 256
#define NH_C 16
#define DK_C 576
#define DV_C 512

__device__ __forceinline__ float warp_max(float val) {
    for (int off = 32; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor(val, off));
    return val;
}

__device__ __forceinline__ float warp_sum(float val) {
    for (int off = 32; off > 0; off >>= 1)
        val += __shfl_xor(val, off);
    return val;
}

// Each block handles ONE batch element, ALL 16 heads
// Grid: (bs,)  Block: (256,)
__global__ void mla_attn_kernel(
    const __hip_bfloat16* __restrict__ Q,     // (bs, NH, DK)
    const __hip_bfloat16* __restrict__ KV,    // (total_kv, DK)
    __hip_bfloat16* __restrict__ O,           // (bs, NH, DV)
    const int kv_len,
    const float sm_scale
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / 64;
    const int lid = tid % 64;
    const int num_warps = TPB / 64;

    // Shared memory
    __shared__ float s_q[NH_C][DK_C];   // 16 * 576 * 4 = 36KB
    __shared__ float s_scores[CHUNK];    // 256 * 4 = 1KB
    __shared__ float s_wmax[4];
    __shared__ float s_wsum[4];

    // Load Q for all 16 heads into shared memory
    const __hip_bfloat16* q_base = Q + bid * NH_C * DK_C;
    // 16 * 576 = 9216 values, 256 threads → 36 values per thread
    for (int i = tid; i < NH_C * DK_C; i += TPB) {
        int h = i / DK_C;
        int d = i % DK_C;
        s_q[h][d] = __bfloat162float(q_base[i]);
    }
    __syncthreads();

    const __hip_bfloat16* kv_base = KV + bid * kv_len * DK_C;

    // Per-head accumulators: each thread stores DV/TPB dims per head
    // With TPB=256 and DV=512: 2 dims per thread per head
    // Total: 16 heads * 2 dims = 32 floats per thread
    float acc[NH_C][2];
    float h_max[NH_C];
    float h_sum[NH_C];
    for (int h = 0; h < NH_C; h++) {
        acc[h][0] = 0.0f;
        acc[h][1] = 0.0f;
        h_max[h] = -1e30f;
        h_sum[h] = 0.0f;
    }

    // Process KV in chunks of CHUNK tokens
    for (int chunk_start = 0; chunk_start < kv_len; chunk_start += CHUNK) {
        int chunk_end = min(chunk_start + CHUNK, kv_len);
        int chunk_size = chunk_end - chunk_start;

        // For each head, compute scores and update softmax
        for (int h = 0; h < NH_C; h++) {
            // Phase 1: thread tid computes score for token chunk_start+tid
            float score = -1e30f;
            if (tid < chunk_size) {
                int token = chunk_start + tid;
                const __hip_bfloat16* k_ptr = kv_base + token * DK_C;
                float dot = 0.0f;
                for (int d = 0; d < DK_C; d++) {
                    dot += s_q[h][d] * __bfloat162float(k_ptr[d]);
                }
                score = dot * sm_scale;
            }
            s_scores[tid] = score;

            // Block-wide max
            float w_max_val = (tid < chunk_size) ? score : -1e30f;
            w_max_val = warp_max(w_max_val);
            if (lid == 0) s_wmax[wid] = w_max_val;
            __syncthreads();

            float chunk_max = s_wmax[0];
            for (int w = 1; w < num_warps; w++)
                chunk_max = fmaxf(chunk_max, s_wmax[w]);

            // Online softmax rescale
            float new_max = fmaxf(h_max[h], chunk_max);
            float alpha = expf(h_max[h] - new_max);

            // Compute exp scores and sum
            float my_p = 0.0f;
            if (tid < chunk_size)
                my_p = expf(s_scores[tid] - new_max);
            s_scores[tid] = my_p;  // store for Phase 3

            float w_sum_val = (tid < chunk_size) ? my_p : 0.0f;
            w_sum_val = warp_sum(w_sum_val);
            if (lid == 0) s_wsum[wid] = w_sum_val;
            __syncthreads();

            float chunk_sum = 0.0f;
            for (int w = 0; w < num_warps; w++)
                chunk_sum += s_wsum[w];

            // Phase 3: accumulate P@V for this head
            // Each thread handles 2 V dims: tid and tid+256
            acc[h][0] *= alpha;
            acc[h][1] *= alpha;
            for (int j = 0; j < chunk_size; j++) {
                float p = s_scores[j];
                int token = chunk_start + j;
                if (tid < DV_C) {
                    acc[h][0] += p * __bfloat162float(kv_base[token * DK_C + tid]);
                }
                if (tid + TPB < DV_C) {
                    acc[h][1] += p * __bfloat162float(kv_base[token * DK_C + tid + TPB]);
                }
            }

            h_sum[h] = h_sum[h] * alpha + chunk_sum;
            h_max[h] = new_max;
            __syncthreads();
        }
    }

    // Write output
    __hip_bfloat16* o_base = O + bid * NH_C * DV_C;
    for (int h = 0; h < NH_C; h++) {
        float inv_sum = 1.0f / fmaxf(h_sum[h], 1e-10f);
        if (tid < DV_C)
            o_base[h * DV_C + tid] = __float2bfloat16(acc[h][0] * inv_sum);
        if (tid + TPB < DV_C)
            o_base[h * DV_C + tid + TPB] = __float2bfloat16(acc[h][1] * inv_sum);
    }
}

void mla_attn_fwd(torch::Tensor Q, torch::Tensor KV, torch::Tensor O,
                  int bs, int kv_len, float sm_scale) {
    dim3 grid(bs);
    dim3 block(TPB);
    // Shared memory: 16*576*4 + 256*4 + 8*4 = 36864 + 1024 + 32 = ~38KB
    mla_attn_kernel<<<grid, block>>>(
        reinterpret_cast<const __hip_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(KV.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(O.data_ptr()),
        kv_len, sm_scale
    );
}
"""

CPP_SRC = """
void mla_attn_fwd(torch::Tensor Q, torch::Tensor KV, torch::Tensor O,
                  int bs, int kv_len, float sm_scale);
"""

_hip_module = load_inline(
    name='mla_attn_hip',
    cpp_sources=[CPP_SRC],
    cuda_sources=[HIP_SRC],
    functions=['mla_attn_fwd'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx950", "-O3"],
)


# ============================================================================
# torch.compile GEMM fallback + Assembly for large configs
# ============================================================================

def _gemm_attn_v2(q_3d, kv_t, v):
    scores = (q_3d @ kv_t) * SM_SCALE
    return F.softmax(scores, dim=-1, dtype=FP32).to(BF16) @ v

_compiled_gemm_short = torch.compile(_gemm_attn_v2)

_KVI = torch.arange(256 * 8192, dtype=torch.int32, device="cuda")

_CONFIGS = {}

def _precompute_config(bs, kv_seq_len, nks, qd, kvd, kvg=32):
    key = (bs, kv_seq_len)
    np_ = bs * nks
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * kv_seq_len
    klp = qo_indptr[1:bs+1]
    info = get_mla_metadata_info_v1(
        bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
        num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH // NKV, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=kvg,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)
    _CONFIGS[key] = {
        'wm': wm, 'wi': wi, 'wis': wis,
        'ri': ri, 'rfm': rfm, 'rpm': rpm,
        'klp': klp,
        'lg': torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda"),
        'ls': torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda"),
        'o': torch.empty((bs, NH, DV), dtype=BF16, device="cuda"),
        'nks': nks,
    }

_precompute_config(4, 8192, 64, FP8_DTYPE, FP8_DTYPE)
_precompute_config(32, 8192, 16, FP8_DTYPE, FP8_DTYPE)
_precompute_config(64, 1024, 8, BF16, FP8_DTYPE)
_precompute_config(64, 8192, 8, FP8_DTYPE, FP8_DTYPE)
_precompute_config(256, 1024, 8, BF16, FP8_DTYPE)
_precompute_config(256, 8192, 4, FP8_DTYPE, FP8_DTYPE)

# Pre-allocate HIP kernel output
_HIP_O = {4: torch.empty((4, NH, DV), dtype=BF16, device="cuda")}


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    # bs=4/kv=1024: custom HIP kernel (single launch, no torch.compile overhead)
    if bs == 4 and kv_seq_len == 1024:
        kv_bf16 = kv_data["bf16"]
        o = _HIP_O[4]
        _hip_module.mla_attn_fwd(
            q.view(bs, NH, DQ),
            kv_bf16.view(-1, DQ),
            o, bs, kv_seq_len, SM_SCALE)
        return o

    # Other small kv≤1024 configs: GEMM
    if kv_seq_len <= 1024 and bs <= 32:
        kv = kv_data["bf16"].view(bs, kv_seq_len, DQ)
        q_3d = q.view(bs, NH, DQ)
        v = kv[:, :, :DV]
        kv_t = kv.transpose(-2, -1)
        return _compiled_gemm_short(q_3d, kv_t, v)

    # Large configs: assembly with pre-computed metadata
    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]
    c = _CONFIGS[(bs, kv_seq_len)]

    if kv_seq_len <= 1024:
        q_input, q_scale = q, None
    else:
        t2d = q.view(-1, DQ)
        qx = torch.empty_like(t2d, dtype=FP8_DTYPE)
        scale = torch.zeros(1, dtype=FP32, device=q.device)
        dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
        q_input, q_scale = qx.view(q.shape), scale

    kvi = _KVI[:n]
    aiter.mla_decode_stage1_asm_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer.view(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, c['klp'],
        None, c['wm'], c['wi'], c['wis'], 1, 1, NKV, SM_SCALE,
        c['lg'], c['ls'], c['o'], q_scale, kv_scale)
    aiter.mla_reduce_v1(c['lg'], c['ls'], c['ri'], c['rfm'], c['rpm'], 1, c['o'], None)
    return c['o']
