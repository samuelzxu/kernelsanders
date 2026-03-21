"""
MLA decode — fused MXFP4 with V dequant from MXFP4 (not bf16).

Key optimization: V loaded from MXFP4 (272 bytes/token) instead of bf16 (1024 bytes).
Total bandwidth: K_mxfp4(306) + V_mxfp4(272) = 578 bytes ≈ same as fp8 assembly (576).
But with 2x FP4 MFMA throughput for Q@K^T.

V dequant approach:
- Load V packed uint8 as (BLOCK_N, 256)
- Extract lo/hi nibbles → two (BLOCK_N, 256) FP4 values
- Convert FP4 E2M1 to f32 via bit manipulation
- Apply E8M0 block-32 scales
- Two separate tl.dot(p, v_lo) and tl.dot(p, v_hi) for even/odd DV positions
- Interleave results at store time
"""
import torch, torch.nn.functional as F, triton, triton.language as tl
from task import input_t, output_t
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

BF16, FP32 = torch.bfloat16, torch.float32
NH, DQ, DV = 16, 576, 512
SM_SCALE = 1.0 / (DQ ** 0.5)
DQ_HALF = DQ // 2   # 288
DV_HALF = DV // 2   # 256
N_SCALES = DQ // 32  # 18
V_SCALES = DV // 32  # 16


@triton.jit
def _fp4_to_f32(nibble):
    """Convert FP4 E2M1 nibble (0-15) to float32.

    FP4 E2M1: 1 sign, 2 exponent, 1 mantissa, bias=1
    Values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    """
    abs_val = nibble & 0x7
    exp = (abs_val >> 1)  # 0-3
    man = (abs_val & 1).to(tl.float32)  # 0.0 or 1.0
    exp_f = exp.to(tl.float32)

    # Normal (exp > 0): 2^(exp-1) * (1 + man*0.5)
    # Subnormal (exp == 0): man * 0.5
    # Compute both and select
    normal_val = tl.exp2(exp_f - 1.0) * (1.0 + man * 0.5)
    subnormal_val = man * 0.5
    result = tl.where(exp > 0, normal_val, subnormal_val)

    # Apply sign
    sign = 1.0 - 2.0 * ((nibble >> 3) & 1).to(tl.float32)
    return result * sign


@triton.jit
def _stage1_mxfp4v(
    Q,          # (bs, NH, DQ) bf16
    KV_fp4,     # (n, DQ_HALF) uint8 — K AND V data
    KV_scale,   # (n, N_SCALES) uint8 — E8M0 scales
    Att_Out,    # (bs, NH, NKS, DV+1) f32
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

    # --- Load Q bf16 and quantize to MXFP4 ---
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

    # KV split range
    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    spl = tl.cdiv(seq_len, NKS)
    s0 = spl * split_id
    s1 = tl.minimum(s0 + spl, seq_len)

    e_max = tl.zeros([PADDED_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([PADDED_H], dtype=tl.float32)
    # Separate accumulators for even and odd DV positions
    acc_lo = tl.zeros([PADDED_H, DV_HALF_C], dtype=tl.float32)  # DV positions 0,2,4,...
    acc_hi = tl.zeros([PADDED_H, DV_HALF_C], dtype=tl.float32)  # DV positions 1,3,5,...

    # K offsets
    offs_k_nope = tl.arange(0, 256)
    offs_k_rope = tl.arange(0, 32)
    offs_sc_nope = tl.arange(0, 16)
    offs_sc_rope = tl.arange(0, 2)
    # V offsets (packed bytes)
    offs_v_bytes = tl.arange(0, DV_HALF_C)  # 0..255
    offs_v_sc = tl.arange(0, 16)  # V_SCALES = 16

    if s1 > s0:
        for start_n in range(s0, s1, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < s1
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=valid, other=0)

            # === Q@K^T via dot_scaled ===
            # K nope: (256, BLOCK_N) = (K_packed, N)
            k_nope = tl.load(
                KV_fp4 + kv_loc[None, :] * stride_kfp4 + offs_k_nope[:, None],
                mask=valid[None, :], other=0,
            )
            k_nope_sc = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + offs_sc_nope[None, :],
                mask=valid[:, None], other=0,
            )
            # K rope: (32, BLOCK_N) = (K_packed, N)
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
            acc_lo *= rescale[:, None]
            acc_hi *= rescale[:, None]

            # === V from MXFP4: dequant and accumulate ===
            # Load V packed bytes: (BLOCK_N, 256)
            v_packed = tl.load(
                KV_fp4 + kv_loc[:, None] * stride_kfp4 + offs_v_bytes[None, :],
                mask=valid[:, None], other=0,
            )
            # Load V scales: (BLOCK_N, 16) E8M0
            v_sc_u8 = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + offs_v_sc[None, :],
                mask=valid[:, None], other=0,
            )

            # Extract lo/hi nibbles
            v_lo_nib = v_packed & 0xF        # (BLOCK_N, 256) — even DV positions
            v_hi_nib = (v_packed >> 4) & 0xF  # (BLOCK_N, 256) — odd DV positions

            # Convert FP4 E2M1 to f32
            v_lo_f32 = _fp4_to_f32(v_lo_nib)  # (BLOCK_N, 256) f32
            v_hi_f32 = _fp4_to_f32(v_hi_nib)  # (BLOCK_N, 256) f32

            # Apply E8M0 scales: each scale covers 16 bytes (32 FP4 values)
            # Scale group for byte i: i // 16
            # v_sc_expanded: (BLOCK_N, 256) — broadcast scale to each byte
            sc_idx = offs_v_bytes[None, :] // 16  # (1, 256) → groups 0..15
            # Gather scales: we need v_sc_u8[:, sc_idx] but can't do 2D gather
            # Instead, compute scale factor per byte position
            v_sc_f32 = tl.exp2(v_sc_u8.to(tl.float32) - 127.0)  # (BLOCK_N, 16)

            # Expand scales from (BLOCK_N, 16) to (BLOCK_N, 256) via repeated load
            # Each group of 16 bytes shares one scale
            # We can do this by loading scales with stride tricks
            # Actually, use reshape: repeat each scale 16 times
            # Triton doesn't have repeat, so load with computed offsets
            sc_per_byte = tl.load(
                KV_scale + kv_loc[:, None] * stride_kscale + (offs_v_bytes[None, :] // 16),
                mask=valid[:, None], other=127,  # 127 → scale=1.0
            )
            sc_expanded = tl.exp2(sc_per_byte.to(tl.float32) - 127.0)  # (BLOCK_N, 256)

            v_lo_scaled = v_lo_f32 * sc_expanded
            v_hi_scaled = v_hi_f32 * sc_expanded

            # p @ V_lo and p @ V_hi
            acc_lo += tl.dot(p.to(tl.bfloat16), v_lo_scaled.to(tl.bfloat16))
            acc_hi += tl.dot(p.to(tl.bfloat16), v_hi_scaled.to(tl.bfloat16))

            e_sum = e_sum * rescale + tl.sum(p, 1)
            e_max = new_max

    # Normalize
    inv_sum = 1.0 / tl.maximum(e_sum, 1e-12)
    acc_lo = acc_lo * inv_sum[:, None]
    acc_hi = acc_hi * inv_sum[:, None]

    # Store interleaved: output[h, 2*i] = acc_lo[h, i], output[h, 2*i+1] = acc_hi[h, i]
    offs_dv_half = tl.arange(0, DV_HALF_C)
    base = cur_batch * stride_ab + heads[:, None] * stride_ah + split_id * stride_as

    # Even DV positions (0, 2, 4, ...)
    tl.store(Att_Out + base + (2 * offs_dv_half[None, :]),
             acc_lo, mask=mask_h[:, None] & (2 * offs_dv_half[None, :] < 512))
    # Odd DV positions (1, 3, 5, ...)
    tl.store(Att_Out + base + (2 * offs_dv_half[None, :] + 1),
             acc_hi, mask=mask_h[:, None] & (2 * offs_dv_half[None, :] + 1 < 512))
    # LSE
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

    # Fused MXFP4 attention with MXFP4 V dequant
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

    _stage1_mxfp4v[(bs * nks,)](
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
