"""
MLA decode — hybrid with HIP MXFP4 V dequant for medium configs.
- Small: torch.compile GEMM
- Medium (bs=32/kv=8192): MXFP4 Triton Q@K^T + HIP V dequant (L2 cache)
- Large (bs>=64): fp8 assembly
"""
import os
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_VERBOSE', '0')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS', '100')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS', '50')

import torch, torch.nn.functional as F, triton, triton.language as tl
import ctypes, urllib.request
from task import input_t, output_t
import aiter
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

FP8_DTYPE = aiter_dtypes.fp8
BF16, FP32 = torch.bfloat16, torch.float32
NH, NKV, DQ, DV = 16, 1, 576, 512
SM_SCALE = 1.0 / (DQ ** 0.5)
DQ_HALF = DQ // 2
DV_HALF = DV // 2

# ─── Download and load precompiled HIP V dequant kernel ───
_SO_PATH = "/tmp/v_dequant_v2_gfx950.so"
_SO_URL = "https://github.com/samuelzxu/gfx950-kernels/releases/download/v0.1/v_dequant_v2_gfx950.so"
_dequant_lib = None

def _get_dequant_lib():
    global _dequant_lib
    if _dequant_lib is not None:
        return _dequant_lib
    try:
        if not os.path.exists(_SO_PATH):
            urllib.request.urlretrieve(_SO_URL, _SO_PATH)
        _dequant_lib = ctypes.CDLL(_SO_PATH)
        _dequant_lib.launch_v_dequant.restype = ctypes.c_int
        _dequant_lib.launch_v_dequant.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p
        ]
        return _dequant_lib
    except Exception:
        return None

# Pre-load at import time
_get_dequant_lib()


def _hip_dequant_v(kv_fp4_2d, kv_sc_2d, n):
    """Dequant V from MXFP4 to bf16 using HIP kernel. Returns bf16 tensor or None."""
    lib = _get_dequant_lib()
    if lib is None:
        return None
    v_out = torch.empty((n, DV), dtype=BF16, device="cuda")
    err = lib.launch_v_dequant(
        ctypes.c_void_p(kv_fp4_2d.data_ptr()),
        ctypes.c_void_p(kv_sc_2d.data_ptr()),
        ctypes.c_void_p(v_out.data_ptr()),
        ctypes.c_int(n),
        ctypes.c_int(DV_HALF),
        ctypes.c_int(DV),
        ctypes.c_int(32),
        ctypes.c_void_p(0)  # default HIP device execution path
    )
    if err != 0:
        return None
    return v_out


# ─── torch.compile GEMM ───
def _gemm_attn(q3, kt, v):
    s = torch.empty(q3.shape[0], NH, kt.shape[2], dtype=BF16, device=q3.device)
    torch.baddbmm(s, q3, kt, beta=0, alpha=SM_SCALE, out=s)
    return torch.bmm(F.softmax(s, dim=-1, dtype=FP32).to(BF16), v)

_cs = torch.compile(_gemm_attn)
_cl = torch.compile(_gemm_attn)


# ─── MXFP4 Triton kernel ───
@triton.jit
def _mxfp4_stage1(
    Q, KV_fp4, KV_scale, V_bf16, Att_Out,
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
    PADDED_H: tl.constexpr = 32
    heads = tl.arange(0, PADDED_H)
    mask_h = heads < BLOCK_H
    SCALE_GRP: tl.constexpr = 32
    offs_nope = tl.arange(0, 512)
    q_nope = tl.load(Q + cur_batch * stride_qb + heads[:, None] * stride_qh + offs_nope[None, :],
                      mask=mask_h[:, None], other=0.0)
    q_nope_e2m1, q_nope_sc = _mxfp4_quant_op(q_nope.to(tl.float32), 512, PADDED_H, SCALE_GRP)
    offs_rope = tl.arange(0, 64)
    q_rope = tl.load(Q + cur_batch * stride_qb + heads[:, None] * stride_qh + (512 + offs_rope[None, :]),
                      mask=mask_h[:, None], other=0.0)
    q_rope_e2m1, q_rope_sc = _mxfp4_quant_op(q_rope.to(tl.float32), 64, PADDED_H, SCALE_GRP)
    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    spl = tl.cdiv(seq_len, NKS)
    s0 = spl * split_id
    s1 = tl.minimum(s0 + spl, seq_len)
    e_max = tl.zeros([PADDED_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([PADDED_H], dtype=tl.float32)
    acc = tl.zeros([PADDED_H, BLOCK_DV], dtype=tl.float32)
    offs_k_nope = tl.arange(0, 256)
    offs_k_rope = tl.arange(0, 32)
    offs_sc_nope = tl.arange(0, 16)
    offs_sc_rope = tl.arange(0, 2)
    offs_v = tl.arange(0, BLOCK_DV)
    if s1 > s0:
        for start_n in range(s0, s1, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid = offs_n < s1
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=valid, other=0)
            k_nope = tl.load(KV_fp4 + kv_loc[None, :] * stride_kfp4 + offs_k_nope[:, None],
                             mask=valid[None, :], other=0)
            k_nope_sc = tl.load(KV_scale + kv_loc[:, None] * stride_kscale + offs_sc_nope[None, :],
                                mask=valid[:, None], other=0)
            k_rope = tl.load(KV_fp4 + kv_loc[None, :] * stride_kfp4 + (256 + offs_k_rope[:, None]),
                             mask=valid[None, :], other=0)
            k_rope_sc = tl.load(KV_scale + kv_loc[:, None] * stride_kscale + (16 + offs_sc_rope[None, :]),
                                mask=valid[:, None], other=0)
            qk = tl.zeros([PADDED_H, BLOCK_N], dtype=tl.float32)
            qk = tl.dot_scaled(q_nope_e2m1, q_nope_sc, "e2m1",
                               k_nope, k_nope_sc, "e2m1", fast_math=True, acc=qk)
            qk = tl.dot_scaled(q_rope_e2m1, q_rope_sc, "e2m1",
                               k_rope, k_rope_sc, "e2m1", fast_math=True, acc=qk)
            qk *= sm_scale
            qk = tl.where(valid[None, :], qk, float("-inf"))
            new_max = tl.maximum(tl.max(qk, 1), e_max)
            rescale = tl.exp(e_max - new_max)
            p = tl.exp(qk - new_max[:, None])
            acc *= rescale[:, None]
            v = tl.load(V_bf16 + kv_loc[:, None] * stride_vb + offs_v[None, :],
                        mask=valid[:, None], other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * rescale + tl.sum(p, 1)
            e_max = new_max
    offs_out = tl.arange(0, BLOCK_DV)
    mid = cur_batch * stride_ab + heads[:, None] * stride_ah + split_id * stride_as + offs_out[None, :]
    tl.store(Att_Out + mid, acc / tl.maximum(e_sum[:, None], 1e-12),
             mask=mask_h[:, None] & (offs_out[None, :] < 512))
    lse_off = cur_batch * stride_ab + heads * stride_ah + split_id * stride_as + 512
    tl.store(Att_Out + lse_off, e_max + tl.log(tl.maximum(e_sum, 1e-12)), mask=mask_h)


@triton.jit
def _mxfp4_stage2(
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


def _run_mxfp4_with_hip_v(q, kv_data, kv_indptr, bs):
    """MXFP4 Q@K^T + HIP-dequanted V from MXFP4."""
    kv_fp4, kv_sc = kv_data["mxfp4"]
    n = kv_fp4.shape[0]
    nks = 16

    kv_fp4_2d = kv_fp4.view(n, DQ_HALF)
    if kv_fp4_2d.dtype != torch.uint8:
        kv_fp4_2d = kv_fp4_2d.view(torch.uint8)
    kv_sc_2d = kv_sc if kv_sc.dtype == torch.uint8 else kv_sc.view(torch.uint8)

    # Dequant V via HIP kernel (reads 272 bytes/token, writes bf16 to GPU mem)
    v_bf16 = _hip_dequant_v(kv_fp4_2d, kv_sc_2d, n)
    if v_bf16 is None:
        # Fallback to bf16 V from data
        v_bf16 = kv_data["bf16"].view(-1, DQ)[:, :DV]

    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    att = torch.empty((bs, NH, nks, DV + 1), dtype=FP32, device="cuda")

    _mxfp4_stage1[(bs * nks,)](
        q, kv_fp4_2d, kv_sc_2d, v_bf16, att, kv_indptr, kvi,
        q.stride(0), q.stride(1),
        kv_fp4_2d.stride(0), kv_sc_2d.stride(0), v_bf16.stride(0),
        att.stride(0), att.stride(1), att.stride(2),
        sm_scale=SM_SCALE,
        BLOCK_N=32, BLOCK_H=NH, NKS=nks, BLOCK_DV=DV, BS=bs,
        num_warps=8, num_stages=2,
    )
    BDV = triton.next_power_of_2(DV)
    _mxfp4_stage2[(bs * NH,)](
        att, o, kv_indptr,
        att.stride(0), att.stride(1), att.stride(2),
        o.stride(0), o.stride(1),
        Lv=DV, HN=NH, BS=bs, NKS=nks, BDV=BDV,
        num_warps=4, num_stages=2,
    )
    return o


def _run_assembly(q, kv_data, qo_indptr, kv_indptr, bs, kv_seq_len):
    """fp8 assembly kernel path."""
    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]
    if kv_seq_len <= 1024:
        q_input, q_scale, qd = q, None, BF16
        kvg, nks = 32, 8
    else:
        t2d = q.view(-1, DQ)
        qx = torch.empty(t2d.shape, dtype=FP8_DTYPE, device=q.device)
        q_scale = torch.empty(1, dtype=FP32, device=q.device)
        aiter.dynamic_per_tensor_quant(qx, t2d, q_scale)
        q_input = qx.view(q.shape)
        qd = FP8_DTYPE
        kvg, nks = 32, 32
    kvd = kv_buffer.dtype
    info = get_mla_metadata_info_v1(
        bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
        num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    klp = qo_indptr[1:bs+1]
    kvi = torch.arange(n, dtype=torch.int32, device="cuda")
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH // NKV, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=kvg,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)
    np_ = bs * nks
    lg = torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda")
    ls = torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
    aiter.mla_decode_stage1_asm_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer.view(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE,
        lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kvl = cfg.get("kv_seq_len", 1024)

    # Small configs: torch.compile GEMM
    if bs <= 4 or (bs <= 32 and kvl <= 1024):
        kv = kv_data["bf16"].view(bs, kvl, DQ)
        q3 = q.view(bs, NH, DQ)
        if kvl <= 1024:
            return _cs(q3, kv.transpose(-2, -1), kv[:, :, :DV])
        return _cl(q3, kv.transpose(-2, -1), kv[:, :, :DV])

    # Large configs: fp8 assembly
    return _run_assembly(q, kv_data, qo_indptr, kv_indptr, bs, kvl)
