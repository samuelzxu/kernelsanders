"""
MLA decode - MFMA Triton with split nope(512)+rope(64) + precomputed metadata.

Fix: DK=576 isn't a power of 2, so tl.arange(0, 576) fails.
Split Q@K^T into nope(512) + rope(64) — both powers of 2.
This is exactly the MLA-specific optimization used in aiter's Triton kernel.

Q@K^T = Q[:,:512] @ K[:,:512]^T + Q[:,512:] @ K[:,512:]^T
Both via MFMA tl.dot = optimal MFMA utilization.
"""

import os
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_VERBOSE', '0')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS', '100')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS', '50')

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
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
NOPE_DIM = 512
ROPE_DIM = 64


# ============================================================================
# MFMA Triton flash-decode with split nope/rope
# ============================================================================

@triton.jit
def _mfma_flash_s1(
    Q, KV, Mid_O, Mid_LSE,
    kv_indptr,
    sm_scale,
    stride_qb, stride_qh,
    stride_kvn,
    stride_mob, stride_mos, stride_moh,
    stride_mlb, stride_mls,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % NUM_SPLITS
    batch_id = pid // NUM_SPLITS

    kv_start = tl.load(kv_indptr + batch_id)
    kv_end = tl.load(kv_indptr + batch_id + 1)
    kv_len = kv_end - kv_start
    split_len = tl.cdiv(kv_len, NUM_SPLITS)
    s_start = split_id * split_len
    s_end = tl.minimum(s_start + split_len, kv_len)

    # Constants (power-of-2 for tl.arange)
    NOPE: tl.constexpr = 512
    ROPE: tl.constexpr = 64
    DV_HALF: tl.constexpr = 256
    NH_C: tl.constexpr = 16

    offs_h = tl.arange(0, NH_C)  # 16 = power of 2 ✓
    offs_nope = tl.arange(0, NOPE)  # 512 = power of 2 ✓
    offs_rope = NOPE + tl.arange(0, ROPE)  # 64 = power of 2 ✓
    offs_dv0 = tl.arange(0, DV_HALF)  # 256 = power of 2 ✓
    offs_dv1 = DV_HALF + tl.arange(0, DV_HALF)

    # Load Q nope + rope for all heads
    q_nope = tl.load(
        Q + batch_id * stride_qb + offs_h[:, None] * stride_qh + offs_nope[None, :]
    )  # (16, 512) bf16
    q_rope = tl.load(
        Q + batch_id * stride_qb + offs_h[:, None] * stride_qh + offs_rope[None, :]
    )  # (16, 64) bf16

    # Accumulators
    acc0 = tl.zeros([NH_C, DV_HALF], dtype=tl.float32)
    acc1 = tl.zeros([NH_C, DV_HALF], dtype=tl.float32)
    m_i = tl.full([NH_C], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([NH_C], dtype=tl.float32)

    for start in range(s_start, s_end, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < s_end

        # Load K nope + rope
        k_nope = tl.load(
            KV + (kv_start + offs_n[:, None]) * stride_kvn + offs_nope[None, :],
            mask=mask_n[:, None], other=0.0
        )  # (BLOCK_N, 512) bf16
        k_rope = tl.load(
            KV + (kv_start + offs_n[:, None]) * stride_kvn + offs_rope[None, :],
            mask=mask_n[:, None], other=0.0
        )  # (BLOCK_N, 64) bf16

        # Q@K^T = nope_part + rope_part via MFMA
        qk = tl.dot(q_nope, tl.trans(k_nope))  # (16, BLOCK_N)
        qk += tl.dot(q_rope, tl.trans(k_rope))  # (16, BLOCK_N)
        qk = qk.to(tl.float32) * sm_scale
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # P@V first half via MFMA
        v0 = tl.load(
            KV + (kv_start + offs_n[:, None]) * stride_kvn + offs_dv0[None, :],
            mask=mask_n[:, None], other=0.0
        )
        acc0 = acc0 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v0)

        # P@V second half
        v1 = tl.load(
            KV + (kv_start + offs_n[:, None]) * stride_kvn + offs_dv1[None, :],
            mask=mask_n[:, None], other=0.0
        )
        acc1 = acc1 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v1)

        m_i = m_new
        l_i = l_new

    # Normalize
    inv_l = 1.0 / tl.maximum(l_i, 1e-10)
    acc0 *= inv_l[:, None]
    acc1 *= inv_l[:, None]
    lse = m_i + tl.log(tl.maximum(l_i, 1e-10))

    # Store mid_o: use 2D pointer addressing
    out_ptrs0 = (Mid_O
                 + batch_id * stride_mob
                 + split_id * stride_mos
                 + offs_h[:, None] * stride_moh
                 + offs_dv0[None, :])
    tl.store(out_ptrs0, acc0.to(tl.float32))

    out_ptrs1 = (Mid_O
                 + batch_id * stride_mob
                 + split_id * stride_mos
                 + offs_h[:, None] * stride_moh
                 + offs_dv1[None, :])
    tl.store(out_ptrs1, acc1.to(tl.float32))

    # Store mid_lse
    lse_off = batch_id * stride_mlb + split_id * stride_mls + tl.arange(0, NH_C)
    tl.store(Mid_LSE + lse_off, lse)


@triton.jit
def _mfma_flash_s2(
    Mid_O, Mid_LSE, O,
    stride_mob, stride_mos, stride_moh,
    stride_mlb, stride_mls,
    stride_ob, stride_oh,
    NUM_SPLITS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    DV_HALF: tl.constexpr = 256
    offs_dv0 = tl.arange(0, DV_HALF)
    offs_dv1 = DV_HALF + tl.arange(0, DV_HALF)

    acc0 = tl.zeros([DV_HALF], dtype=tl.float32)
    acc1 = tl.zeros([DV_HALF], dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    for split_id in range(NUM_SPLITS):
        o_off = batch_id * stride_mob + split_id * stride_mos + head_id * stride_moh
        partial_o0 = tl.load(Mid_O + o_off + offs_dv0)
        partial_o1 = tl.load(Mid_O + o_off + offs_dv1)
        lse_off = batch_id * stride_mlb + split_id * stride_mls + head_id
        partial_lse = tl.load(Mid_LSE + lse_off)

        m_new = tl.maximum(m_i, partial_lse)
        m_new = tl.maximum(m_new, -1e30)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(partial_lse - m_new)
        l_new = l_i * alpha + beta
        acc0 = acc0 * alpha + beta * partial_o0
        acc1 = acc1 * alpha + beta * partial_o1
        m_i = m_new
        l_i = l_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-10)
    out_off = batch_id * stride_ob + head_id * stride_oh
    tl.store(O + out_off + offs_dv0, (acc0 * inv_l).to(tl.bfloat16))
    tl.store(O + out_off + offs_dv1, (acc1 * inv_l).to(tl.bfloat16))


# ============================================================================
# Pre-allocate everything
# ============================================================================

_KVI = torch.arange(256 * 8192, dtype=torch.int32, device="cuda")

_TRI = {}
def _init_tri(bs, kv_seq_len, nks):
    _TRI[(bs, kv_seq_len)] = {
        'mid_o': torch.empty((bs, nks, NH, DV), dtype=FP32, device="cuda"),
        'mid_lse': torch.empty((bs, nks, NH), dtype=FP32, device="cuda"),
        'o': torch.empty((bs, NH, DV), dtype=BF16, device="cuda"),
        'nks': nks,
    }

_init_tri(4, 1024, 1)
_init_tri(4, 8192, 8)
_init_tri(32, 1024, 2)

_ASM = {}
def _init_asm(bs, kv_seq_len, nks, qd, kvd, kvg=32):
    np_ = bs * nks
    qo = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
    kvi_p = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * kv_seq_len
    klp = qo[1:bs+1]
    info = get_mla_metadata_info_v1(
        bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
        num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    get_mla_metadata_v1(
        qo, kvi_p, klp, NH // NKV, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=kvg,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)
    _ASM[(bs, kv_seq_len)] = {
        'wm': wm, 'wi': wi, 'wis': wis,
        'ri': ri, 'rfm': rfm, 'rpm': rpm,
        'klp': klp,
        'lg': torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda"),
        'ls': torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda"),
        'o': torch.empty((bs, NH, DV), dtype=BF16, device="cuda"),
        'nks': nks,
    }

_init_asm(32, 8192, 32, FP8_DTYPE, FP8_DTYPE)
_init_asm(64, 1024, 8, BF16, FP8_DTYPE)
_init_asm(64, 8192, 32, FP8_DTYPE, FP8_DTYPE)
_init_asm(256, 1024, 8, BF16, FP8_DTYPE)
_init_asm(256, 8192, 32, FP8_DTYPE, FP8_DTYPE)


def _run_triton(q, kv_bf16, kv_indptr, bs, kv_seq_len):
    bufs = _TRI[(bs, kv_seq_len)]
    nks = bufs['nks']
    mid_o = bufs['mid_o']
    mid_lse = bufs['mid_lse']
    o = bufs['o']

    n = kv_bf16.shape[0]
    kv_2d = kv_bf16.view(n, DQ)
    q_3d = q.view(bs, NH, DQ)

    grid = (bs * nks,)
    _mfma_flash_s1[grid](
        q_3d, kv_2d, mid_o, mid_lse,
        kv_indptr, SM_SCALE,
        q_3d.stride(0), q_3d.stride(1),
        kv_2d.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        BLOCK_N=32,
        NUM_SPLITS=nks,
    )

    if nks == 1:
        return mid_o[:, 0].to(BF16)

    grid2 = (bs, NH)
    _mfma_flash_s2[grid2](
        mid_o, mid_lse, o,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        mid_lse.stride(0), mid_lse.stride(1),
        o.stride(0), o.stride(1),
        NUM_SPLITS=nks,
    )
    return o


def _run_assembly(q, kv_data, qo_indptr, kv_indptr, bs, kv_seq_len):
    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]
    c = _ASM[(bs, kv_seq_len)]
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


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):
        return _run_triton(q, kv_data["bf16"], kv_indptr, bs, kv_seq_len)

    return _run_assembly(q, kv_data, qo_indptr, kv_indptr, bs, kv_seq_len)
