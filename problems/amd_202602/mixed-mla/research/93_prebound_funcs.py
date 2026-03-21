"""
MLA decode - honest, minimized Python overhead.
a16w8 for kv<=1024, a8w8+Triton quant for kv>1024.
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.triton.quant.quant import dynamic_per_tensor_quant_fp8_i8

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
I32 = torch.int32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)

# Pre-bind frequently called functions to avoid attribute lookups
_empty = torch.empty
_ones = torch.ones
_arange = torch.arange
_stage1 = aiter.mla_decode_stage1_asm_fwd
_reduce = aiter.mla_reduce_v1


def _quant(tensor):
    t2d = tensor.view(-1, DQ)
    qx = _empty(t2d.shape, dtype=FP8_DTYPE, device=tensor.device)
    scale = torch.zeros(1, dtype=FP32, device=tensor.device)
    dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
    return qx.view(tensor.shape), scale


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kvl = cfg.get("kv_seq_len", 1024)
    kvb, kvs = kv_data["fp8"]
    n = kvb.shape[0]
    dev = q.device
    nks = 16 if bs <= 4 else 32

    if kvl <= 1024:
        qi, qs, qd, kvg = q, None, BF16, 16
    else:
        qi, qs = _quant(q)
        qd, kvg = FP8_DTYPE, 64

    kvd = kvb.dtype
    info = get_mla_metadata_info_v1(
        bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
        num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [_empty(s, dtype=t, device=dev) for s, t in info]

    klp = _ones(bs, dtype=I32, device=dev)
    kvi = _arange(n, dtype=I32, device=dev)

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=kvg,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)

    np_ = bs * nks
    lg = _empty((np_, 1, NH, DV), dtype=FP32, device=dev)
    ls = _empty((np_, 1, NH, 1), dtype=FP32, device=dev)
    o = _empty((bs, NH, DV), dtype=BF16, device=dev)

    _stage1(qi.view(-1, NH, DQ), kvb.view(n, 1, NKV, DQ),
            qo_indptr, kv_indptr, kvi, klp,
            None, wm, wi, wis, 1, 1, NKV, SM_SCALE,
            lg, ls, o, qs, kvs)
    _reduce(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
