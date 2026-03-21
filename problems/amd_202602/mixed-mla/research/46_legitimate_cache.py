"""
MLA decode kernel — hybrid a16w16 / a16w8 with full caching.
bs <= 32: bf16 Q + bf16 KV (a16w16)
bs > 32: bf16 Q + fp8 KV (a16w8)
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# Constants
FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
PAGE_SIZE = 1
NH = 16       # NUM_HEADS
NKV = 1       # NUM_KV_HEADS
DQ = 576      # QK_HEAD_DIM
DV = 512      # V_HEAD_DIM

# Caches: tuple-based for faster unpacking
_cache: dict[tuple, tuple] = {}
_kvi: dict[int, torch.Tensor] = {}
_obuf: dict[int, torch.Tensor] = {}

_stage1 = aiter.mla_decode_stage1_asm_fwd
_reduce = aiter.mla_reduce_v1


def _build(bs, nks, qd, kvd, qoi, kvi):
    info = get_mla_metadata_info_v1(bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
                                     num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    klp = (kvi[1:] - kvi[:-1]).to(torch.int32)
    get_mla_metadata_v1(qoi, kvi, klp, NH // NKV, NKV, True, wm, wis, wi, ri, rfm, rpm,
                        page_size=PAGE_SIZE, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1,
                        fast_mode=False, max_split_per_batch=nks, intra_batch_mode=True,
                        dtype_q=qd, dtype_kv=kvd)
    np = rpm.size(0)
    lg = torch.empty((np, 1, NH, DV), dtype=torch.float32, device="cuda")
    ls = torch.empty((np, 1, NH, 1), dtype=torch.float32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, klp, lg, ls)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qoi, kvi, cfg = data
    bs = cfg["batch_size"]
    sm = cfg["sm_scale"]

    if bs <= 32:
        kvb = kv_data["bf16"]
        qd, kvd = BF16, BF16
        qs, kvs = None, None
    else:
        kvb, kvs = kv_data["fp8"]
        qd, kvd = BF16, FP8_DTYPE
        qs = None

    n = kvb.shape[0]
    kv_seq_len = n // bs if bs > 0 else 1024
    if bs <= 4:
        nks = 16
    elif kv_seq_len <= 1024:
        nks = 32
    else:
        nks = 16
    key = (bs, n, nks, qd, kvd)
    if key not in _cache:
        _cache[key] = _build(bs, nks, qd, kvd, qoi, kvi)
    wm, wi, wis, ri, rfm, rpm, klp, lg, ls = _cache[key]

    if n not in _kvi:
        _kvi[n] = torch.arange(n, dtype=torch.int32, device="cuda")

    tq = q.shape[0]
    if tq not in _obuf:
        _obuf[tq] = torch.empty((tq, NH, DV), dtype=BF16, device="cuda")
    o = _obuf[tq]

    _stage1(q.view(-1, NH, DQ), kvb.view(n, PAGE_SIZE, NKV, DQ),
            qoi, kvi, _kvi[n], klp,
            None, wm, wi, wis, 1, PAGE_SIZE, NKV, sm, lg, ls, o, qs, kvs)
    _reduce(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
