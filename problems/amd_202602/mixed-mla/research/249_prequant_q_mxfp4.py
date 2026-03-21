"""
MLA decode — fused MXFP4 attention with PRE-QUANTIZED Q.

Key fix: quantize Q to MXFP4 OUTSIDE the kernel using downcast_to_mxfp(),
then pass pre-quantized Q to the Triton kernel. This avoids inlining
_mxfp4_quant_op which caused 17-minute compilation timeout.

K from kv_data["mxfp4"], V from kv_data["bf16"][:,:,:512].
tl.dot_scaled for Q@K^T (FP4×FP4 MFMA), tl.dot for attn@V (bf16).
"""
import torch, torch.nn.functional as F, triton, triton.language as tl
from task import input_t, output_t
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

BF16, FP32 = torch.bfloat16, torch.float32
NH, DQ, DV = 16, 576, 512
SM_SCALE = 1.0 / (DQ ** 0.5)
DQ_HALF = DQ // 2
N_K_SCALES = DQ // 32  # 18
SCALE_GROUP = 32


@triton.jit
def _mla_mxfp4_prequant_stage1(
    Q_fp4,      # (bs, NH, DQ//2) uint8 — pre-quantized Q
    Q_scales,   # (bs, NH, N_K_SCALES) float32 — Q scales
    KV_fp4,     # (total_kv, DQ_HALF) uint8
    KV_scale,   # (total_kv_pad, N_K_SCALES) uint8/e8m0
    V_bf16,     # (total_kv, DV) bf16
    Att_Out,    # (bs, NH, NKS, DV+1) fp32
    kv_indptr, kv_indices,
    stride_qb, stride_qh,
    stride_qsb, stride_qsh,
    stride_kfp4, stride_kscale, stride_vb,
    stride_ab, stride_ah, stride_as,
    sm_scale: tl.constexpr,
    ADQ: tl.constexpr, ADV: tl.constexpr, ADQ_HALF: tl.constexpr,
    NKSC: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    NKS: tl.constexpr, BLOCK_DV: tl.constexpr, BS: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = pid % NKS
    cur_batch = pid // NKS
    if cur_batch >= BS:
        return

    heads = tl.arange(0, BLOCK_H)

    # Load pre-quantized Q (already in MXFP4 format)
    # Pad to power of 2: 288 → 512, 18 → 32
    PADDED_DQ_HALF: tl.constexpr = 256  # nope part only (512 dims = 256 packed)
    PADDED_NKSC: tl.constexpr = 32
    offs_q = tl.arange(0, PADDED_DQ_HALF)
    q_e2m1 = tl.load(
        Q_fp4 + cur_batch * stride_qb + heads[:, None] * stride_qh + offs_q[None, :],
        mask=offs_q[None, :] < ADQ_HALF, other=0,
    )  # (BLOCK_H, 512) with last 224 elements = 0

    # Load Q scales
    offs_qs = tl.arange(0, PADDED_NKSC)
    q_scales = tl.load(
        Q_scales + cur_batch * stride_qsb + heads[:, None] * stride_qsh + offs_qs[None, :],
        mask=offs_qs[None, :] < NKSC, other=0.0,
    )  # (BLOCK_H, 32) with last 14 elements = 0

    # KV split range
    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    spl = tl.cdiv(seq_len, NKS)
    s0 = spl * split_id
    s1 = tl.minimum(s0 + spl, seq_len)

    # Accumulators
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    offs_k = tl.arange(0, PADDED_DQ_HALF)  # 512, padded from 288
    offs_ks = tl.arange(0, PADDED_NKSC)   # 32, padded from 18
    offs_v = tl.arange(0, BLOCK_DV)       # 512, already power of 2

    if s1 > s0:
        for start_n in range(s0, s1, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < s1
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=valid, other=0)

            # Load K MXFP4
            k_fp4 = tl.load(
                KV_fp4 + kv_loc[:, None] * stride_kfp4 + offs_k[None, :],
                mask=valid[:, None] & (offs_k[None, :] < ADQ_HALF), other=0,
            )
            k_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + offs_ks[None, :],
                mask=valid[:, None] & (offs_ks[None, :] < NKSC), other=0,
            )

            # Q@K^T via FP4 MFMA — both Q and K already in MXFP4
            qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
            qk = tl.dot_scaled(
                q_e2m1, q_scales, "e2m1",
                k_fp4, k_sc, "e2m1",
                fast_math=True, acc=qk,
            )
            qk *= sm_scale
            qk = tl.where(valid[None, :], qk, float("-inf"))

            # Online softmax
            new_max = tl.maximum(tl.max(qk, 1), e_max)
            rescale = tl.exp(e_max - new_max)
            p = tl.exp(qk - new_max[:, None])
            acc *= rescale[:, None]

            # Load V bf16
            v = tl.load(
                V_bf16 + kv_loc[:, None] * stride_vb + offs_v[None, :],
                mask=valid[:, None] & (offs_v[None, :] < ADV), other=0.0,
            )
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(p, 1)
            e_max = new_max

    # Store output + lse
    offs_out = tl.arange(0, BLOCK_DV)
    mid = cur_batch * stride_ab + heads[:, None] * stride_ah + split_id * stride_as + offs_out[None, :]
    tl.store(Att_Out + mid, acc / tl.maximum(e_sum[:, None], 1e-12),
             mask=offs_out[None, :] < ADV)
    lse_off = cur_batch * stride_ab + heads * stride_ah + split_id * stride_as + ADV
    tl.store(Att_Out + lse_off, e_max + tl.log(tl.maximum(e_sum, 1e-12)))


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

    # Compiled GEMM for small configs
    if bs <= 4 or (bs <= 32 and kvl <= 1024):
        kv = kv_data["bf16"].view(bs, kvl, DQ)
        q3 = q.view(bs, NH, DQ)
        if kvl <= 1024:
            return _cs(q3, kv.transpose(-2, -1), kv[:, :, :DV])
        return _cl(q3, kv.transpose(-2, -1), kv[:, :, :DV])

    # Fused MXFP4 attention for large configs
    kv_fp4, kv_sc = kv_data["mxfp4"]
    v_bf16 = kv_data["bf16"].view(-1, DQ)[:, :DV]
    n = kv_fp4.shape[0]
    nks = 16

    # Pre-quantize Q to MXFP4 OUTSIDE the kernel (avoids _mxfp4_quant_op in JIT)
    q_2d = q.view(-1, DQ)  # (bs, 576) bf16 (since total_q = bs for decode)
    q_fp4_raw, q_scale_raw = downcast_to_mxfp(q_2d, torch.uint8, axis=-1)
    # Ensure uint8 dtype for Triton compatibility
    q_fp4 = q_fp4_raw if q_fp4_raw.dtype == torch.uint8 else q_fp4_raw.view(torch.uint8)
    q_fp4_3d = q_fp4.view(bs, NH, DQ_HALF)
    # Scales: ensure float32
    q_sc = q_scale_raw if q_scale_raw.dtype == FP32 else q_scale_raw.to(FP32)
    q_scales_3d = q_sc.view(bs, NH, N_K_SCALES)

    # KV data
    kv_fp4_2d = kv_fp4.view(n, DQ_HALF)
    if kv_fp4_2d.dtype != torch.uint8:
        kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_sc_2d = kv_sc
    if kv_sc_2d.dtype != torch.uint8:
        kv_sc_2d = kv_sc_2d.view(torch.uint8)

    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    att = torch.empty((bs, NH, nks, DV + 1), dtype=FP32, device="cuda")

    _mla_mxfp4_prequant_stage1[(bs * nks,)](
        q_fp4_3d, q_scales_3d,
        kv_fp4_2d, kv_sc_2d, v_bf16, att,
        kv_indptr, kvi,
        q_fp4_3d.stride(0), q_fp4_3d.stride(1),
        q_scales_3d.stride(0), q_scales_3d.stride(1),
        kv_fp4_2d.stride(0), kv_sc_2d.stride(0), v_bf16.stride(0),
        att.stride(0), att.stride(1), att.stride(2),
        sm_scale=SM_SCALE, ADQ=DQ, ADV=DV, ADQ_HALF=DQ_HALF,
        NKSC=N_K_SCALES, BLOCK_N=32, BLOCK_H=NH,
        NKS=nks, BLOCK_DV=DV, BS=bs,
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
