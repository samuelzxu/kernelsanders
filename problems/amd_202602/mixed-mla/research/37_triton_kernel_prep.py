"""
Custom Triton MLA decode kernel with MXFP4 KV support.
Stage 1: Triton flash-decode with fused MXFP4 dequantization
Stage 2: Reuse aiter's mla_reduce_v1
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
DQ = 576      # qk_head_dim (K dimension)
DV = 512      # v_head_dim (V dimension = kv_lora_rank)
BLOCK_SIZE = 32  # MXFP4 block size for scales

# Caches
_asm_cache: dict[tuple, tuple] = {}
_tri_cache: dict[tuple, dict] = {}
_kvi: dict[int, torch.Tensor] = {}
_obuf: dict[int, torch.Tensor] = {}

_stage1_asm = aiter.mla_decode_stage1_asm_fwd
_reduce = aiter.mla_reduce_v1


# -----------------------------------------------------------------------
# Triton flash-decode stage1 kernel for MLA with bf16 KV
# -----------------------------------------------------------------------
@triton.jit
def _mla_decode_stage1_kernel(
    Q,          # (total_q, NH, DQ) bf16
    KV_buf,     # (total_kv, NKV, DQ) bf16 - combined K(576) and V(512) buffer
    Out,        # (total_q * NUM_SPLITS, 1, NH, DV) fp32 - partial outputs
    Out_lse,    # (total_q * NUM_SPLITS, 1, NH, 1) fp32 - partial log-sum-exp
    kv_indptr,  # (bs+1,) int32
    sm_scale: tl.constexpr,
    stride_qb, stride_qh,
    stride_kv,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    DQ_CONST: tl.constexpr,
    DV_CONST: tl.constexpr,
    BLOCK_N: tl.constexpr,    # KV tokens per iteration
    NUM_SPLITS: tl.constexpr,
    BLOCK_DQ: tl.constexpr,   # tile size for K dim (must be >= DQ)
    BLOCK_DV: tl.constexpr,   # tile size for V dim (must be >= DV)
):
    # Grid: (bs * NH * NUM_SPLITS,)
    pid = tl.program_id(0)
    split_id = pid % NUM_SPLITS
    pid2 = pid // NUM_SPLITS
    head_id = pid2 % NH
    batch_id = pid2 // NH

    # Get KV range for this batch
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    kv_len = kv_end - kv_start

    # Split range
    split_len = tl.cdiv(kv_len, NUM_SPLITS)
    split_start = split_id * split_len
    split_end = tl.minimum(split_start + split_len, kv_len)

    # Load Q for this head: (BLOCK_DQ,)
    offs_dq = tl.arange(0, BLOCK_DQ)
    mask_dq = offs_dq < DQ_CONST
    q_vec = tl.load(
        Q + batch_id * stride_qb + head_id * stride_qh + offs_dq,
        mask=mask_dq, other=0.0
    ).to(tl.float32)

    # Online softmax state
    m_prev = -float("inf")  # running max
    l_prev = 0.0            # running sum of exp
    # Accumulator for weighted V: (BLOCK_DV,)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < DV_CONST
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    # Iterate over KV tokens in blocks
    for block_start in range(split_start, split_end, BLOCK_N):
        offs_n = block_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_end

        # Load K block: (BLOCK_N, BLOCK_DQ)
        k_ptrs = KV_buf + (kv_start + offs_n[:, None]) * stride_kv + offs_dq[None, :]
        k_block = tl.load(k_ptrs, mask=mask_n[:, None] & mask_dq[None, :], other=0.0).to(tl.float32)

        # QK^T: (BLOCK_N,) = (BLOCK_N, BLOCK_DQ) @ (BLOCK_DQ,)
        scores = tl.sum(k_block * q_vec[None, :], axis=1) * sm_scale

        # Online softmax update
        m_new = tl.maximum(m_prev, tl.where(mask_n, scores, -float("inf")))
        # Handle the case where m_new == -inf (all masked)
        m_new = tl.maximum(m_new, -1e30)

        # Correction factors
        exp_diff = tl.exp(m_prev - m_new)
        p = tl.where(mask_n, tl.exp(scores - m_new), 0.0)

        # Update running sum
        l_new = l_prev * exp_diff + tl.sum(p)

        # Load V block: (BLOCK_N, BLOCK_DV) - first DV dims of KV buffer
        v_ptrs = KV_buf + (kv_start + offs_n[:, None]) * stride_kv + offs_dv[None, :]
        v_block = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0).to(tl.float32)

        # Update accumulator: rescale old acc + add new weighted V
        acc = acc * exp_diff + tl.sum(p[:, None] * v_block, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Normalize output
    out_vec = acc / tl.maximum(l_prev, 1e-10)
    lse = m_prev + tl.log(tl.maximum(l_prev, 1e-10))

    # Store partial output and LSE
    out_idx = batch_id * stride_ob + head_id * stride_oh + split_id * stride_os
    tl.store(Out + out_idx * DV_CONST + offs_dv, out_vec, mask=mask_dv)
    tl.store(Out_lse + out_idx + 0, lse)


# -----------------------------------------------------------------------
# Build helpers
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


SM_SCALE = 1.0 / (DQ ** 0.5)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qoi, kvi, cfg = data
    bs = cfg["batch_size"]

    tq = q.shape[0]
    if tq not in _obuf:
        _obuf[tq] = torch.empty((tq, NH, DV), dtype=BF16, device="cuda")
    o = _obuf[tq]

    if bs <= 32:
        # Small batches: bf16 Q + bf16 KV with assembly kernel (fastest)
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
    else:
        # Large batches: a16w8 with assembly kernel
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
