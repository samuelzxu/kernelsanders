"""
MLA decode — fused MXFP4 with split nope+rope, _mxfp4_quant_op inside kernel.

FIX: Load K data as (K_packed, BLOCK_N) not (BLOCK_N, K_packed).
tl.dot_scaled expects: LHS=(M,K), RHS=(K,N), LHS_scale=(M,K//32), RHS_scale=(N,K//32).

Split 576-dim into nope(512) + rope(64): all dims are powers of 2.
Q quantized on-the-fly via _mxfp4_quant_op (produces MFMA-compatible scales).
K from kv_data["mxfp4"]. V from bf16.
PADDED_H=32 for MFMA tile requirement.
"""
import torch, torch.nn.functional as F, triton, triton.language as tl
from task import input_t, output_t
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

BF16, FP32 = torch.bfloat16, torch.float32
NH, DQ, DV = 16, 576, 512
SM_SCALE = 1.0 / (DQ ** 0.5)
NOPE_DIM, ROPE_DIM = 512, 64
NOPE_HALF, ROPE_HALF = 256, 32
NOPE_SCALES, ROPE_SCALES = 16, 2
DQ_HALF = DQ // 2   # 288
SCALE_GROUP = 32


@triton.jit
def _stage1(
    Q,          # (bs, NH, DQ) bf16
    KV_fp4,     # (n, DQ_HALF) uint8
    KV_scale,   # (n_pad, N_SCALES) uint8
    V_bf16,     # (n, DV) bf16
    Att_Out,    # (bs, NH, NKS, DV+1) f32
    kv_indptr, kv_indices,
    stride_qb, stride_qh, stride_kfp4, stride_kscale, stride_vb,
    stride_ab, stride_ah, stride_as,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    NKS: tl.constexpr, BLOCK_DV: tl.constexpr, BS: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % NKS
    cur_batch = pid // NKS
    if cur_batch >= BS:
        return

    # Pad H from 16 to 32 for MFMA tile requirement
    PADDED_H: tl.constexpr = 32
    heads = tl.arange(0, PADDED_H)
    mask_h = heads < BLOCK_H

    # --- Load Q bf16 and quantize to MXFP4 via _mxfp4_quant_op ---
    # Nope part: Q[:, :512] → (PADDED_H, 512) bf16
    offs_nope = tl.arange(0, 512)  # NOPE_DIM = 512, power of 2!
    q_nope_bf16 = tl.load(
        Q + cur_batch * stride_qb + heads[:, None] * stride_qh + offs_nope[None, :],
        mask=mask_h[:, None], other=0.0,
    )
    SCALE_GRP: tl.constexpr = 32
    q_nope_e2m1, q_nope_sc = _mxfp4_quant_op(
        q_nope_bf16.to(tl.float32), 512, PADDED_H, SCALE_GRP
    )
    # q_nope_e2m1: (PADDED_H=32, 256) — LHS for dot_scaled
    # q_nope_sc: (PADDED_H=32, 16) — LHS scales

    # Rope part: Q[:, 512:576] → (PADDED_H, 64) bf16
    offs_rope = tl.arange(0, 64)  # ROPE_DIM = 64, power of 2!
    q_rope_bf16 = tl.load(
        Q + cur_batch * stride_qb + heads[:, None] * stride_qh + (512 + offs_rope[None, :]),
        mask=mask_h[:, None], other=0.0,
    )
    q_rope_e2m1, q_rope_sc = _mxfp4_quant_op(
        q_rope_bf16.to(tl.float32), 64, PADDED_H, SCALE_GRP
    )
    # q_rope_e2m1: (PADDED_H=32, 32) — LHS for dot_scaled
    # q_rope_sc: (PADDED_H=32, 2) — LHS scales

    # KV split range
    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    spl = tl.cdiv(seq_len, NKS)
    s0 = spl * split_id
    s1 = tl.minimum(s0 + spl, seq_len)

    e_max = tl.zeros([PADDED_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([PADDED_H], dtype=tl.float32)
    acc = tl.zeros([PADDED_H, BLOCK_DV], dtype=tl.float32)

    # Offsets for K loading — TRANSPOSED: (K_packed, BLOCK_N) layout for RHS
    offs_k_nope = tl.arange(0, 256)   # NOPE_HALF = 256 (K dim, rows)
    offs_k_rope = tl.arange(0, 32)    # ROPE_HALF = 32 (K dim, rows)
    # Scales: (BLOCK_N, num_scales) layout — NOT transposed
    offs_sc_nope = tl.arange(0, 16)   # NOPE_SCALES = 16
    offs_sc_rope = tl.arange(0, 2)    # ROPE_SCALES = 2 (already power of 2)
    offs_v = tl.arange(0, BLOCK_DV)   # 512

    if s1 > s0:
        for start_n in range(s0, s1, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < s1
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=valid, other=0)

            # --- Load K nope MXFP4: (256, BLOCK_N) = (K_packed, N) ---
            # Transposed loading: K dim as rows, N as columns
            k_nope = tl.load(
                KV_fp4 + kv_loc[None, :] * stride_kfp4 + offs_k_nope[:, None],
                mask=valid[None, :], other=0,
            )
            # K nope scales: (BLOCK_N, 16) = (N, K//32)
            k_nope_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + offs_sc_nope[None, :],
                mask=valid[:, None], other=0,
            )

            # --- Load K rope MXFP4: (32, BLOCK_N) = (K_packed, N) ---
            k_rope = tl.load(
                KV_fp4 + kv_loc[None, :] * stride_kfp4 + (256 + offs_k_rope[:, None]),
                mask=valid[None, :], other=0,
            )
            # K rope scales: (BLOCK_N, 2) = (N, K//32)
            k_rope_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + (16 + offs_sc_rope[None, :]),
                mask=valid[:, None], other=0,
            )

            # --- Q@K^T = nope_score + rope_score ---
            qk = tl.zeros([PADDED_H, BLOCK_N], dtype=tl.float32)

            # Nope: LHS=(32, 256) @ RHS=(256, 32) → (32, 32)
            qk = tl.dot_scaled(
                q_nope_e2m1, q_nope_sc, "e2m1",
                k_nope, k_nope_sc, "e2m1",
                fast_math=True, acc=qk,
            )

            # Rope: LHS=(32, 32) @ RHS=(32, 32) → (32, 32)
            qk = tl.dot_scaled(
                q_rope_e2m1, q_rope_sc, "e2m1",
                k_rope, k_rope_sc, "e2m1",
                fast_math=True, acc=qk,
            )

            qk *= sm_scale
            qk = tl.where(valid[None, :], qk, float("-inf"))

            # Online softmax
            new_max = tl.maximum(tl.max(qk, 1), e_max)
            rescale = tl.exp(e_max - new_max)
            p = tl.exp(qk - new_max[:, None])
            acc *= rescale[:, None]

            # V bf16: (BLOCK_N, 512)
            v = tl.load(
                V_bf16 + kv_loc[:, None] * stride_vb + offs_v[None, :],
                mask=valid[:, None], other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(p, 1)
            e_max = new_max

    # Store — only first BLOCK_H=16 heads
    offs_out = tl.arange(0, BLOCK_DV)
    mid = cur_batch * stride_ab + heads[:, None] * stride_ah + split_id * stride_as + offs_out[None, :]
    tl.store(Att_Out + mid, acc / tl.maximum(e_sum[:, None], 1e-12),
             mask=mask_h[:, None] & (offs_out[None, :] < 512))
    lse_off = cur_batch * stride_ab + heads * stride_ah + split_id * stride_as + 512
    tl.store(Att_Out + lse_off, e_max + tl.log(tl.maximum(e_sum, 1e-12)), mask=mask_h)


@triton.jit
def _stage2(
    L, O, kv_indptr, stride_lb, stride_lh, stride_ls, stride_ob, stride_oh,
    Lv: tl.constexpr, HN: tl.constexpr, BS: tl.constexpr,
    NKS: tl.constexpr, BDV: tl.constexpr,
):
    pid = tl.program_id(0)
    b, h = pid // HN, pid % HN
    if b >= BS:
        return
    seq = tl.load(kv_indptr + b + 1) - tl.load(kv_indptr + b)
    spl = tl.cdiv(seq, NKS)
    d = tl.arange(0, BDV)
    md = d < Lv
    es, em, ac = 0.0, -float("inf"), tl.zeros([BDV], dtype=tl.float32)
    for s in range(NKS):
        if spl * s >= seq:
            continue
        tv = tl.load(L + b*stride_lb + h*stride_lh + s*stride_ls + d, mask=md, other=0.0)
        tl_ = tl.load(L + b*stride_lb + h*stride_lh + s*stride_ls + Lv)
        nm = tl.maximum(tl_, em)
        os_ = tl.exp(em - nm)
        ac *= os_
        el = tl.exp(tl_ - nm)
        ac += el * tv
        es = es * os_ + el
        em = nm
    tl.store(O + b*stride_ob + h*stride_oh + d,
             (ac / tl.maximum(es, 1e-12)).to(O.dtype.element_ty), mask=md)


def _gemm_attn(q3, kt, v):
    bs, kl = q3.shape[0], kt.shape[2]
    s = torch.empty(bs, NH, kl, dtype=BF16, device=q3.device)
    torch.baddbmm(s, q3, kt, beta=0, alpha=SM_SCALE, out=s)
    return torch.bmm(F.softmax(s, dim=-1, dtype=FP32).to(BF16), v)

_cs = torch.compile(_gemm_attn)
_cl = torch.compile(_gemm_attn)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kvl = cfg.get("kv_seq_len", 1024)

    if bs <= 4 or (bs <= 32 and kvl <= 1024):
        kv = kv_data["bf16"].view(bs, kvl, DQ)
        q3 = q.view(bs, NH, DQ)
        if kvl <= 1024:
            return _cs(q3, kv.transpose(-2, -1), kv[:, :, :DV])
        return _cl(q3, kv.transpose(-2, -1), kv[:, :, :DV])

    # Fused MXFP4 attention
    kv_fp4, kv_sc = kv_data["mxfp4"]
    v_bf16 = kv_data["bf16"].view(-1, DQ)[:, :DV]
    n = kv_fp4.shape[0]
    nks = 16

    kv_fp4_2d = kv_fp4.view(n, DQ_HALF)
    if kv_fp4_2d.dtype != torch.uint8:
        kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_sc_2d = kv_sc if kv_sc.dtype == torch.uint8 else kv_sc.view(torch.uint8)

    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    att = torch.empty((bs, NH, nks, DV + 1), dtype=FP32, device="cuda")

    _stage1[(bs * nks,)](
        q, kv_fp4_2d, kv_sc_2d, v_bf16, att, kv_indptr, kvi,
        q.stride(0), q.stride(1),
        kv_fp4_2d.stride(0), kv_sc_2d.stride(0), v_bf16.stride(0),
        att.stride(0), att.stride(1), att.stride(2),
        sm_scale=SM_SCALE,
        BLOCK_N=32, BLOCK_H=NH, NKS=nks, BLOCK_DV=DV, BS=bs,
        num_warps=8, num_stages=2,
    )

    BDV = triton.next_power_of_2(DV)
    _stage2[(bs * NH,)](
        att, o, kv_indptr,
        att.stride(0), att.stride(1), att.stride(2),
        o.stride(0), o.stride(1),
        Lv=DV, HN=NH, BS=bs, NKS=nks, BDV=BDV,
        num_warps=4, num_stages=2,
    )
    return o
