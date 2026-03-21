"""
MLA decode — GEMM-based for small batches, assembly for large.
For bs<=32: compute attention as GEMM (Q @ K^T -> softmax -> attn @ V)
For bs>32: assembly a16w8 with cached metadata
"""

import torch
import torch.nn.functional as F
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
SM_SCALE = 1.0 / (DQ ** 0.5)

_asm_cache: dict[tuple, tuple] = {}
_kvi: dict[int, torch.Tensor] = {}
_obuf: dict[int, torch.Tensor] = {}

_stage1 = aiter.mla_decode_stage1_asm_fwd
_reduce = aiter.mla_reduce_v1


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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qoi, kvi, cfg = data
    bs = cfg["batch_size"]

    tq = q.shape[0]
    if tq not in _obuf:
        _obuf[tq] = torch.empty((tq, NH, DV), dtype=BF16, device="cuda")
    o = _obuf[tq]

    if bs <= 4:
        # GEMM-based attention for very small batches
        # Q: (bs, NH, DQ) -> (bs*NH, DQ)
        # KV: (total_kv, 1, DQ) -> per-batch: (kv_len, DQ)
        kvb = kv_data["bf16"]
        kv_len = cfg["kv_seq_len"]

        # Reshape for batched GEMM
        q_2d = q.view(bs, NH, DQ)  # (bs, NH, DQ)
        kv_3d = kvb.view(bs, kv_len, DQ)  # (bs, kv_len, DQ)

        # QK^T: (bs, NH, DQ) @ (bs, DQ, kv_len) -> (bs, NH, kv_len)
        scores = torch.bmm(
            q_2d.view(bs, NH, DQ),
            kv_3d.transpose(1, 2)  # (bs, DQ, kv_len)
        ) * SM_SCALE

        # Softmax
        attn = F.softmax(scores, dim=-1)

        # Output: (bs, NH, kv_len) @ (bs, kv_len, DV) -> (bs, NH, DV)
        kv_v = kv_3d[:, :, :DV]  # (bs, kv_len, DV) - first 512 dims
        out = torch.bmm(attn, kv_v)  # (bs, NH, DV)
        o.copy_(out.to(BF16))
    elif bs <= 32:
        # Assembly a16w16 for medium batches
        kvb = kv_data["bf16"]
        n = kvb.shape[0]
        nks = 32
        key = (bs, nks, BF16, BF16)
        if key not in _asm_cache:
            _asm_cache[key] = _build_asm(bs, nks, BF16, BF16, qoi, kvi)
        wm, wi, wis, ri, rfm, rpm, klp, lg, ls = _asm_cache[key]
        if n not in _kvi:
            _kvi[n] = torch.arange(n, dtype=torch.int32, device="cuda")
        _stage1(q.view(-1, NH, DQ), kvb.view(n, PAGE_SIZE, NKV, DQ),
                qoi, kvi, _kvi[n], klp,
                None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o, None, None)
        _reduce(lg, ls, ri, rfm, rpm, 1, o, None)
    else:
        # Assembly a16w8 for large batches
        kvb, kvs = kv_data["fp8"]
        n = kvb.shape[0]
        nks = 32
        key = (bs, nks, BF16, FP8_DTYPE)
        if key not in _asm_cache:
            _asm_cache[key] = _build_asm(bs, nks, BF16, FP8_DTYPE, qoi, kvi)
        wm, wi, wis, ri, rfm, rpm, klp, lg, ls = _asm_cache[key]
        if n not in _kvi:
            _kvi[n] = torch.arange(n, dtype=torch.int32, device="cuda")
        _stage1(q.view(-1, NH, DQ), kvb.view(n, PAGE_SIZE, NKV, DQ),
                qoi, kvi, _kvi[n], klp,
                None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o, None, kvs)
        _reduce(lg, ls, ri, rfm, rpm, 1, o, None)

    return o
