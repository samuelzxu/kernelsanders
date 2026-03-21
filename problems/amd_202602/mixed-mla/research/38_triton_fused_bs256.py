"""
Custom Triton MLA decode kernel for large batches (bs=256) using bf16 KV.
Fused stage1+reduce in a single Triton kernel to eliminate reduce overhead.
Assembly kernels for smaller batches.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# Constants
FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
PAGE_SIZE = 1
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)

# Caches
_asm_cache: dict[tuple, tuple] = {}
_kvi: dict[int, torch.Tensor] = {}
_obuf: dict[int, torch.Tensor] = {}

_stage1_asm = aiter.mla_decode_stage1_asm_fwd
_reduce = aiter.mla_reduce_v1


# -----------------------------------------------------------------------
# Triton MLA decode: single-pass flash-decode per (batch, head)
# No split-K, no reduce needed. Each thread block does one head of one batch.
# -----------------------------------------------------------------------
@triton.jit
def _mla_decode_fused_kernel(
    Q,          # (bs, NH, DQ) bf16
    KV_buf,     # (total_kv, 1, DQ) bf16
    Out,        # (bs, NH, DV) bf16
    kv_indptr,  # (bs+1,) int32
    sm_scale,
    stride_qbs, stride_qh,
    stride_kv_token,
    stride_obs, stride_oh,
    BLOCK_N: tl.constexpr,    # KV tokens per iteration (e.g., 128)
    BLOCK_DK: tl.constexpr,   # must be >= 576
    BLOCK_DV: tl.constexpr,   # must be >= 512
):
    # Each program handles one (batch, head) pair
    pid = tl.program_id(0)
    head_id = pid % NH
    batch_id = pid // NH

    # KV range for this batch
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    kv_len = kv_end - kv_start

    # Load Q: (BLOCK_DK,)
    offs_dk = tl.arange(0, BLOCK_DK)
    mask_dk = offs_dk < 576
    q_vec = tl.load(
        Q + batch_id * stride_qbs + head_id * stride_qh + offs_dk,
        mask=mask_dk, other=0.0
    ).to(tl.float32)

    # Output accumulator: (BLOCK_DV,)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < 512
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    # Stream through KV tokens
    for start in range(0, kv_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len

        # Load K: (BLOCK_N, BLOCK_DK)
        k_ptrs = KV_buf + (kv_start + offs_n[:, None]) * stride_kv_token + offs_dk[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_dk[None, :], other=0.0).to(tl.float32)

        # Scores: (BLOCK_N,) = sum over d of k[n,d] * q[d]
        s = tl.sum(k * q_vec[None, :], axis=1) * sm_scale

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(tl.where(mask_n, s, -float("inf"))))
        m_new = tl.maximum(m_new, -1e30)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(mask_n, tl.exp(s - m_new), 0.0)
        l_new = l_i * alpha + tl.sum(p)

        # Load V: (BLOCK_N, BLOCK_DV) - first 512 dims
        v_ptrs = KV_buf + (kv_start + offs_n[:, None]) * stride_kv_token + offs_dv[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0).to(tl.float32)

        # Update acc
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    # Normalize and store
    out_vec = (acc / tl.maximum(l_i, 1e-10)).to(tl.bfloat16)
    out_ptr = Out + batch_id * stride_obs + head_id * stride_oh + offs_dv
    tl.store(out_ptr, out_vec, mask=mask_dv)


# -----------------------------------------------------------------------
# Assembly kernel helpers
# -----------------------------------------------------------------------
def _build_asm(bs, nks, qd, kvd, qoi, kvi):
    info = get_mla_metadata_info_v1(bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
                                     num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    klp = (kvi[1:] - kvi[:-1]).to(torch.int32)
    get_mla_metadata_v1(qoi, kvi, klp, NH // NKV, NKV, True, wm, wis, wi, ri, rfm, rpm,
                        page_size=PAGE_SIZE, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1,
                        fast_mode=False, max_split_per_batch=nks, intra_batch_mode=True,
                        dtype_q=qd, dtype_kv=kvd)
    np_ = rpm.size(0)
    lg = torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda")
    ls = torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, klp, lg, ls)


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------
# Threshold for using Triton kernel
USE_TRITON_BS = 256  # enable Triton for bs >= 256

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qoi, kvi, cfg = data
    bs = cfg["batch_size"]

    tq = q.shape[0]
    if tq not in _obuf:
        _obuf[tq] = torch.empty((tq, NH, DV), dtype=BF16, device="cuda")
    o = _obuf[tq]

    if bs <= 32:
        # Small batches: a16w16 assembly
        kvb = kv_data["bf16"]
        n = kvb.shape[0]
        nks = 16 if bs <= 4 else 32
        key = (bs, nks, BF16, BF16)
        if key not in _asm_cache:
            _asm_cache[key] = _build_asm(bs, nks, BF16, BF16, qoi, kvi)
        wm, wi, wis, ri, rfm, rpm, klp, lg, ls = _asm_cache[key]
        if n not in _kvi:
            _kvi[n] = torch.arange(n, dtype=torch.int32, device="cuda")
        _stage1_asm(q.view(-1, NH, DQ), kvb.view(n, PAGE_SIZE, NKV, DQ),
                    qoi, kvi, _kvi[n], klp,
                    None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o, None, None)
        _reduce(lg, ls, ri, rfm, rpm, 1, o, None)
    elif bs >= USE_TRITON_BS:
        # Large batches: custom Triton (fused, no reduce needed)
        kvb = kv_data["bf16"]
        n = kvb.shape[0]
        grid = (bs * NH,)
        kv_2d = kvb.view(n, DQ)
        _mla_decode_fused_kernel[grid](
            q.view(-1, NH, DQ), kv_2d, o, kvi, SM_SCALE,
            q.view(-1, NH, DQ).stride(0), q.view(-1, NH, DQ).stride(1),
            kv_2d.stride(0),
            o.stride(0), o.stride(1),
            BLOCK_N=128, BLOCK_DK=1024, BLOCK_DV=512,
        )
    else:
        # Medium/large batches: a16w8 assembly
        kvb, kvs = kv_data["fp8"]
        n = kvb.shape[0]
        nks = 32
        key = (bs, nks, BF16, FP8_DTYPE)
        if key not in _asm_cache:
            _asm_cache[key] = _build_asm(bs, nks, BF16, FP8_DTYPE, qoi, kvi)
        wm, wi, wis, ri, rfm, rpm, klp, lg, ls = _asm_cache[key]
        if n not in _kvi:
            _kvi[n] = torch.arange(n, dtype=torch.int32, device="cuda")
        _stage1_asm(q.view(-1, NH, DQ), kvb.view(n, PAGE_SIZE, NKV, DQ),
                    qoi, kvi, _kvi[n], klp,
                    None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o, None, kvs)
        _reduce(lg, ls, ri, rfm, rpm, 1, o, None)

    return o
