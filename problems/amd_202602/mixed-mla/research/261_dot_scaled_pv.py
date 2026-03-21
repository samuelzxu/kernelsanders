"""
MLA decode — attempt to use tl.dot_scaled for p@V with rhs_k_pack=False.

If this works, V is loaded from MXFP4 (272 bytes/token) with hardware dequant.
Total bandwidth: K_mxfp4(306) + V_mxfp4(272) = 578 bytes ≈ fp8.

Key insight: rhs_k_pack=False means packing is along N (output/DV) dimension,
which matches V's natural MXFP4 storage (2 FP4 values per byte along DV).

Shapes for dot_scaled p@V:
- LHS (p quantized): (H=32, K_packed=BLOCK_N/2) e2m1, lhs_k_pack=True
  → K = BLOCK_N
- RHS (V mxfp4): (K=BLOCK_N, N_packed=DV/2=256) e2m1, rhs_k_pack=False
  → N = DV/2 * 2 = DV = 512
- Output: (H=32, DV=512)

Scale requirements:
- LHS scale: (H, K//32) = (32, BLOCK_N//32)
- RHS scale: (N, K//32) = (512, BLOCK_N//32) — PROBLEMATIC: V scales are (BLOCK_N, 16)
  → Need to pass rhs_scale=None and handle scaling separately, OR accept unscaled result
"""
import torch, torch.nn.functional as F, triton, triton.language as tl
from task import input_t, output_t
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

BF16, FP32 = torch.bfloat16, torch.float32
NH, DQ, DV = 16, 576, 512
SM_SCALE = 1.0 / (DQ ** 0.5)
DQ_HALF = DQ // 2
DV_HALF = DV // 2


@triton.jit
def _stage1_pv_scaled(
    Q, KV_fp4, KV_scale, Att_Out,
    kv_indptr, kv_indices,
    stride_qb, stride_qh, stride_kfp4, stride_kscale,
    stride_ab, stride_ah, stride_as,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    NKS: tl.constexpr, BS: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % NKS
    cur_batch = pid // NKS
    if cur_batch >= BS:
        return

    PADDED_H: tl.constexpr = 32
    DV_HALF_C: tl.constexpr = 256
    heads = tl.arange(0, PADDED_H)
    mask_h = heads < BLOCK_H
    SCALE_GRP: tl.constexpr = 32

    # Q quantize
    offs_nope = tl.arange(0, 512)
    q_nope = tl.load(
        Q + cur_batch * stride_qb + heads[:, None] * stride_qh + offs_nope[None, :],
        mask=mask_h[:, None], other=0.0,
    )
    q_nope_e2m1, q_nope_sc = _mxfp4_quant_op(q_nope.to(tl.float32), 512, PADDED_H, SCALE_GRP)

    offs_rope = tl.arange(0, 64)
    q_rope = tl.load(
        Q + cur_batch * stride_qb + heads[:, None] * stride_qh + (512 + offs_rope[None, :]),
        mask=mask_h[:, None], other=0.0,
    )
    q_rope_e2m1, q_rope_sc = _mxfp4_quant_op(q_rope.to(tl.float32), 64, PADDED_H, SCALE_GRP)

    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    spl = tl.cdiv(seq_len, NKS)
    s0 = spl * split_id
    s1 = tl.minimum(s0 + spl, seq_len)

    e_max = tl.zeros([PADDED_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([PADDED_H], dtype=tl.float32)
    acc = tl.zeros([PADDED_H, DV], dtype=tl.float32)  # Full DV=512

    offs_k_nope = tl.arange(0, 256)
    offs_k_rope = tl.arange(0, 32)
    offs_sc_nope = tl.arange(0, 16)
    offs_sc_rope = tl.arange(0, 2)
    offs_v_bytes = tl.arange(0, DV_HALF_C)

    if s1 > s0:
        for start_n in range(s0, s1, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < s1
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=valid, other=0)

            # === Q@K^T via dot_scaled ===
            k_nope = tl.load(
                KV_fp4 + kv_loc[None, :] * stride_kfp4 + offs_k_nope[:, None],
                mask=valid[None, :], other=0,
            )
            k_nope_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + offs_sc_nope[None, :],
                mask=valid[:, None], other=0,
            )
            k_rope = tl.load(
                KV_fp4 + kv_loc[None, :] * stride_kfp4 + (256 + offs_k_rope[:, None]),
                mask=valid[None, :], other=0,
            )
            k_rope_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + (16 + offs_sc_rope[None, :]),
                mask=valid[:, None], other=0,
            )

            qk = tl.zeros([PADDED_H, BLOCK_N], dtype=tl.float32)
            qk = tl.dot_scaled(q_nope_e2m1, q_nope_sc, "e2m1",
                               k_nope, k_nope_sc, "e2m1", fast_math=True, acc=qk)
            qk = tl.dot_scaled(q_rope_e2m1, q_rope_sc, "e2m1",
                               k_rope, k_rope_sc, "e2m1", fast_math=True, acc=qk)
            qk *= sm_scale
            qk = tl.where(valid[None, :], qk, float("-inf"))

            # Online softmax
            new_max = tl.maximum(tl.max(qk, 1), e_max)
            rescale = tl.exp(e_max - new_max)
            p = tl.exp(qk - new_max[:, None])
            acc *= rescale[:, None]

            # === p@V using dot_scaled with rhs_k_pack=False ===
            # Quantize p to MXFP4
            p_e2m1, p_sc = _mxfp4_quant_op(p, BLOCK_N, PADDED_H, SCALE_GRP)
            # p_e2m1: (32, BLOCK_N/2) packed, p_sc: (32, BLOCK_N//32)

            # Load V from MXFP4: (BLOCK_N, DV/2=256) — natural storage layout
            v_fp4 = tl.load(
                KV_fp4 + kv_loc[:, None] * stride_kfp4 + offs_v_bytes[None, :],
                mask=valid[:, None], other=0,
            )
            # V scales: pass None for now (accept unscaled result, apply scales later)
            # TODO: proper scale handling

            # dot_scaled with rhs_k_pack=False:
            # LHS: (32, BLOCK_N/2) e2m1, lhs_k_pack=True → K = BLOCK_N
            # RHS: (BLOCK_N, 256) e2m1, rhs_k_pack=False → K=BLOCK_N, N=512
            acc += tl.dot_scaled(
                p_e2m1, p_sc, "e2m1",
                v_fp4, None, "e2m1",
                fast_math=True,
                rhs_k_pack=False,
            )

            e_sum = e_sum * rescale + tl.sum(p, 1)
            e_max = new_max

    # Store
    offs_out = tl.arange(0, DV)
    mid = cur_batch * stride_ab + heads[:, None] * stride_ah + split_id * stride_as + offs_out[None, :]
    tl.store(Att_Out + mid, acc / tl.maximum(e_sum[:, None], 1e-12),
             mask=mask_h[:, None] & (offs_out[None, :] < DV))
    lse_off = cur_batch * stride_ab + heads * stride_ah + split_id * stride_as + DV
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
        if spl * s < seq:
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
    s = torch.empty(q3.shape[0], NH, kt.shape[2], dtype=BF16, device=q3.device)
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

    # Fused MXFP4 attention — K AND V both from MXFP4
    kv_fp4, kv_sc = kv_data["mxfp4"]
    n = kv_fp4.shape[0]
    nks = 16

    kv_fp4_2d = kv_fp4.view(n, DQ_HALF)
    if kv_fp4_2d.dtype != torch.uint8:
        kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_sc_2d = kv_sc if kv_sc.dtype == torch.uint8 else kv_sc.view(torch.uint8)

    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    att = torch.empty((bs, NH, nks, DV + 1), dtype=FP32, device="cuda")

    _stage1_pv_scaled[(bs * nks,)](
        q, kv_fp4_2d, kv_sc_2d, att, kv_indptr, kvi,
        q.stride(0), q.stride(1),
        kv_fp4_2d.stride(0), kv_sc_2d.stride(0),
        att.stride(0), att.stride(1), att.stride(2),
        sm_scale=SM_SCALE,
        BLOCK_N=32, BLOCK_H=NH, NKS=nks, BS=bs,
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
