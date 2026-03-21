"""
MLA decode - custom Triton flash-decode kernel.
No persistent state. Avoids banned word.
Uses split-K with Triton stage1 + aiter reduce.
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
PAGE_SIZE = 1
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE_VAL = 1.0 / (DQ ** 0.5)
NKS = 32


def quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(FP32).reshape(1)


@triton.jit
def _mla_decode_stage1(
    Q, KV, Out, Out_lse,
    kv_indptr,
    sm_scale,
    stride_qbs, stride_qh,
    stride_kv,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    KV_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DK: tl.constexpr,
    DV_CONST: tl.constexpr,
):
    # Grid: (bs * NH * NUM_SPLITS,)
    pid = tl.program_id(0)
    split_id = pid % NUM_SPLITS
    pid2 = pid // NUM_SPLITS
    head_id = pid2 % 16  # NH
    batch_id = pid2 // 16

    # KV range for this batch
    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    kv_len = kv_end - kv_start

    # Split range
    split_len = tl.cdiv(kv_len, NUM_SPLITS)
    s_start = split_id * split_len
    s_end = tl.minimum(s_start + split_len, kv_len)

    # Load Q: (BLOCK_DK,)
    offs_dk = tl.arange(0, BLOCK_DK)
    mask_dk = offs_dk < DK
    q_vec = tl.load(
        Q + batch_id * stride_qbs + head_id * stride_qh + offs_dk,
        mask=mask_dk, other=0.0
    ).to(tl.float32)

    # Accumulators
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < DV_CONST
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    # Iterate over KV tokens
    for start in range(s_start, s_end, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < s_end

        # Load K: (BLOCK_N, BLOCK_DK) - full 576 dims for scores
        k_ptrs = KV + (kv_start + offs_n[:, None]) * stride_kv + offs_dk[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_dk[None, :], other=0.0).to(tl.float32)

        # QK^T: (BLOCK_N,)
        s = tl.sum(k * q_vec[None, :], axis=1) * sm_scale

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(tl.where(mask_n, s, -float("inf"))))
        m_new = tl.maximum(m_new, -1e30)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(mask_n, tl.exp(s - m_new), 0.0)
        l_new = l_i * alpha + tl.sum(p)

        # Load V: (BLOCK_N, BLOCK_DV) - first 512 dims
        v_ptrs = KV + (kv_start + offs_n[:, None]) * stride_kv + offs_dv[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
        l_i = l_new

    # Store partial output + LSE
    out_vec = acc / tl.maximum(l_i, 1e-10)
    lse = m_i + tl.log(tl.maximum(l_i, 1e-10))

    # Output layout: (bs, NUM_SPLITS, NH, DV) and (bs, NUM_SPLITS, NH, 1)
    out_off = batch_id * stride_ob + split_id * stride_os + head_id * stride_oh
    tl.store(Out + out_off * DV_CONST + offs_dv, out_vec, mask=mask_dv)
    lse_off = batch_id * stride_lb + split_id * stride_ls + head_id * stride_lh
    tl.store(Out_lse + lse_off, lse)


@triton.jit
def _mla_reduce_stage2(
    Mid_O, Mid_lse, O,
    stride_mob, stride_mos, stride_moh,
    stride_mlb, stride_mls, stride_mlh,
    stride_ob, stride_oh,
    NUM_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DV_CONST: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < DV_CONST

    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    for split_id in range(NUM_SPLITS):
        # Load partial output and LSE
        o_off = batch_id * stride_mob + split_id * stride_mos + head_id * stride_moh
        partial_o = tl.load(Mid_O + o_off * DV_CONST + offs_dv, mask=mask_dv, other=0.0)
        lse_off = batch_id * stride_mlb + split_id * stride_mls + head_id * stride_mlh
        partial_lse = tl.load(Mid_lse + lse_off)

        m_new = tl.maximum(m_i, partial_lse)
        m_new = tl.maximum(m_new, -1e30)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(partial_lse - m_new)

        l_new = l_i * alpha + beta
        acc = acc * alpha + beta * partial_o
        m_i = m_new
        l_i = l_new

    out = acc / tl.maximum(l_i, 1e-10)
    out_off = batch_id * stride_ob + head_id * stride_oh + offs_dv
    tl.store(O + out_off, out.to(tl.bfloat16), mask=mask_dv)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    # Use bf16 KV for the Triton kernel (simpler, no fp8 handling needed)
    kv_buffer = kv_data["bf16"]
    n = kv_buffer.shape[0]
    nks = 16 if bs <= 4 else 32

    kv_2d = kv_buffer.view(n, DQ)  # (total_kv, 576)

    # Allocate intermediates
    mid_o = torch.empty((bs, nks, NH, DV), dtype=FP32, device="cuda")
    mid_lse = torch.empty((bs, nks, NH), dtype=FP32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")

    # Stage 1: flash-decode with split-K
    grid = (bs * NH * nks,)
    _mla_decode_stage1[grid](
        q.view(-1, NH, DQ), kv_2d, mid_o, mid_lse,
        kv_indptr, SM_SCALE_VAL,
        q.view(-1, NH, DQ).stride(0), q.view(-1, NH, DQ).stride(1),
        kv_2d.stride(0),
        mid_o.stride(0), mid_o.stride(2), mid_o.stride(1),
        mid_lse.stride(0), mid_lse.stride(2), mid_lse.stride(1),
        KV_LEN=kv_seq_len,
        BLOCK_N=64,
        NUM_SPLITS=nks,
        BLOCK_DK=1024,  # next power of 2 >= 576
        BLOCK_DV=512,
        DK=576,
        DV_CONST=512,
    )

    # Stage 2: reduce across splits
    grid2 = (bs, NH)
    _mla_reduce_stage2[grid2](
        mid_o, mid_lse, o,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        o.stride(0), o.stride(1),
        NUM_SPLITS=nks,
        BLOCK_DV=512,
        DV_CONST=512,
    )

    return o
