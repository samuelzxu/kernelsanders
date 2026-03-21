"""
MLA decode — fused MXFP4 with split nope(512)+rope(64) dot_scaled.

Splits 576-dim Q@K^T into two tl.dot_scaled calls:
  1. nope: Q[:512] @ K[:512] — 256 packed bytes, 16 scale groups (all power of 2)
  2. rope: Q[512:576] @ K[512:576] — 32 packed bytes, 2→4 padded scale groups

Q pre-quantized outside kernel via downcast_to_mxfp.
K from kv_data["mxfp4"]. V from bf16.
"""
import torch, torch.nn.functional as F, triton, triton.language as tl
from task import input_t, output_t
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

BF16, FP32 = torch.bfloat16, torch.float32
NH, DQ, DV = 16, 576, 512
SM_SCALE = 1.0 / (DQ ** 0.5)
NOPE_DIM = 512
ROPE_DIM = 64
NOPE_HALF = NOPE_DIM // 2   # 256 (power of 2!)
ROPE_HALF = ROPE_DIM // 2   # 32 (power of 2!)
NOPE_SCALES = NOPE_DIM // 32  # 16 (power of 2!)
ROPE_SCALES = ROPE_DIM // 32  # 2 → pad to 4
DQ_HALF = DQ // 2  # 288
N_SCALES = DQ // 32  # 18


@triton.jit
def _stage1(
    Q_fp4, Q_scales,  # pre-quantized Q: (bs*NH, DQ_HALF) uint8, (bs*NH, N_SCALES) f32
    KV_fp4, KV_scale,  # MXFP4 K: (n, DQ_HALF) uint8, (n, N_SCALES) uint8
    V_bf16,             # bf16 V: (n, DV) bf16
    Att_Out,            # (bs, NH, NKS, DV+1) f32
    kv_indptr, kv_indices,
    stride_q, stride_qs, stride_kfp4, stride_kscale, stride_vb,
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

    # MFMA dot_scaled requires M >= 32. Pad BLOCK_H=16 to 32 with masking.
    PADDED_H: tl.constexpr = 32
    heads = tl.arange(0, PADDED_H)
    mask_h = heads < BLOCK_H
    q_row_base = cur_batch * BLOCK_H  # row in the flattened Q (bs*NH, ...)

    # Load Q nope part: (PADDED_H=32, 256) packed — extra 16 rows zero-padded
    offs_nope = tl.arange(0, 256)
    q_nope = tl.load(
        Q_fp4 + (q_row_base + heads[:, None]) * stride_q + offs_nope[None, :],
        mask=mask_h[:, None], other=0,
    )

    # Load Q rope part: (PADDED_H=32, 32) packed
    offs_rope = tl.arange(0, 32)
    q_rope = tl.load(
        Q_fp4 + (q_row_base + heads[:, None]) * stride_q + (256 + offs_rope[None, :]),
        mask=mask_h[:, None], other=0,
    )

    # Load Q nope scales: (PADDED_H=32, 16) — uint8 E8M0 format
    offs_nope_sc = tl.arange(0, 16)
    q_nope_sc = tl.load(
        Q_scales + (q_row_base + heads[:, None]) * stride_qs + offs_nope_sc[None, :],
        mask=mask_h[:, None], other=0,
    )

    # Load Q rope scales: (PADDED_H=32, 4) padded from 2 — uint8 E8M0
    offs_rope_sc = tl.arange(0, 4)
    q_rope_sc = tl.load(
        Q_scales + (q_row_base + heads[:, None]) * stride_qs + (16 + offs_rope_sc[None, :]),
        mask=mask_h[:, None] & (offs_rope_sc[None, :] < 2), other=0,
    )

    # KV split range
    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    spl = tl.cdiv(seq_len, NKS)
    s0 = spl * split_id
    s1 = tl.minimum(s0 + spl, seq_len)

    e_max = tl.zeros([PADDED_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([PADDED_H], dtype=tl.float32)
    acc = tl.zeros([PADDED_H, BLOCK_DV], dtype=tl.float32)

    offs_v = tl.arange(0, BLOCK_DV)

    if s1 > s0:
        for start_n in range(s0, s1, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < s1
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=valid, other=0)

            # Load K nope MXFP4: (BLOCK_N, 256) packed
            k_nope = tl.load(
                KV_fp4 + kv_loc[:, None] * stride_kfp4 + offs_nope[None, :],
                mask=valid[:, None], other=0,
            )
            # Load K nope scales: (BLOCK_N, 16)
            k_nope_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + offs_nope_sc[None, :],
                mask=valid[:, None], other=0,
            )

            # Load K rope MXFP4: (BLOCK_N, 32) packed
            k_rope = tl.load(
                KV_fp4 + kv_loc[:, None] * stride_kfp4 + (256 + offs_rope[None, :]),
                mask=valid[:, None], other=0,
            )
            # Load K rope scales: (BLOCK_N, 4) padded from 2
            k_rope_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + (16 + offs_rope_sc[None, :]),
                mask=valid[:, None] & (offs_rope_sc[None, :] < 2), other=0,
            )

            # Split dot product: nope + rope
            qk = tl.zeros([PADDED_H, BLOCK_N], dtype=tl.float32)

            # Nope: (16, 256) @ (32, 256)^T using FP4 MFMA
            qk = tl.dot_scaled(
                q_nope, q_nope_sc, "e2m1",
                k_nope, k_nope_sc, "e2m1",
                fast_math=True, acc=qk,
            )

            # Rope: (16, 32) @ (32, 32)^T using FP4 MFMA
            qk = tl.dot_scaled(
                q_rope, q_rope_sc, "e2m1",
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

            # V bf16
            v = tl.load(
                V_bf16 + kv_loc[:, None] * stride_vb + offs_v[None, :],
                mask=valid[:, None] & (offs_v[None, :] < 512), other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(p, 1)
            e_max = new_max

    # Store — only first BLOCK_H=16 heads (mask out padded rows)
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

    # Pre-quantize Q to MXFP4
    q_2d = q.view(bs * NH, DQ)
    q_fp4_raw, q_scale_raw = downcast_to_mxfp(q_2d, torch.uint8, axis=-1)
    q_fp4 = q_fp4_raw if q_fp4_raw.dtype == torch.uint8 else q_fp4_raw.view(torch.uint8)
    # Keep scales as uint8 (E8M0 format required by tl.dot_scaled)
    q_sc = q_scale_raw if q_scale_raw.dtype == torch.uint8 else q_scale_raw.view(torch.uint8)

    # KV data
    kv_fp4_2d = kv_fp4.view(n, DQ_HALF)
    if kv_fp4_2d.dtype != torch.uint8:
        kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_sc_2d = kv_sc if kv_sc.dtype == torch.uint8 else kv_sc.view(torch.uint8)

    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    att = torch.empty((bs, NH, nks, DV + 1), dtype=FP32, device="cuda")

    _stage1[(bs * nks,)](
        q_fp4, q_sc,
        kv_fp4_2d, kv_sc_2d, v_bf16, att,
        kv_indptr, kvi,
        q_fp4.stride(0), q_sc.stride(0),
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
