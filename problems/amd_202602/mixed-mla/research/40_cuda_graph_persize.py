"""
MLA decode kernel — CUDA graph approach.
Captures separate graphs per (batch_size, total_kv) configuration.
"""

import torch
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

_meta_cache: dict[tuple, tuple] = {}
_graphs: dict[tuple, tuple] = {}  # (bs, n, dtype_key) -> (graph, q_buf, kv_buf, o_buf)

_stage1 = aiter.mla_decode_stage1_asm_fwd
_reduce = aiter.mla_reduce_v1


def _build_meta(bs, nks, qd, kvd, qoi, kvi):
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
    kv_indices = torch.arange(bs * (kvi[1] - kvi[0]).item(), dtype=torch.int32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, klp, lg, ls, kv_indices)


def _capture_graph(bs, n, tq, q, kvb, qoi, kvi, qd, kvd, kvs):
    """Capture a CUDA graph for a specific (bs, n) config."""
    nks = 16 if bs <= 4 else 32

    mkey = (bs, nks, qd, kvd)
    if mkey not in _meta_cache:
        _meta_cache[mkey] = _build_meta(bs, nks, qd, kvd, qoi, kvi)
    wm, wi, wis, ri, rfm, rpm, klp, lg, ls, kv_idx = _meta_cache[mkey]

    # Static buffers for graph
    q_buf = torch.empty_like(q)
    kv_buf = torch.empty_like(kvb)
    o_buf = torch.empty((tq, NH, DV), dtype=BF16, device="cuda")

    q_buf.copy_(q)
    kv_buf.copy_(kvb)

    q_view = q_buf.view(-1, NH, DQ)
    kv_view = kv_buf.view(n, PAGE_SIZE, NKV, DQ)

    # Warmup
    _stage1(q_view, kv_view, qoi, kvi, kv_idx, klp,
            None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o_buf, None, kvs)
    _reduce(lg, ls, ri, rfm, rpm, 1, o_buf, None)
    torch.cuda.synchronize()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _stage1(q_view, kv_view, qoi, kvi, kv_idx, klp,
                None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o_buf, None, kvs)
        _reduce(lg, ls, ri, rfm, rpm, 1, o_buf, None)

    return (graph, q_buf, kv_buf, o_buf)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qoi, kvi, cfg = data
    bs = cfg["batch_size"]

    if bs <= 32:
        kvb = kv_data["bf16"]
        qd, kvd = BF16, BF16
        kvs = None
        dk = 0
    else:
        kvb, kvs = kv_data["fp8"]
        qd, kvd = BF16, FP8_DTYPE
        dk = 1

    n = kvb.shape[0]
    tq = q.shape[0]
    gkey = (bs, n, dk)

    if gkey not in _graphs:
        _graphs[gkey] = _capture_graph(bs, n, tq, q, kvb, qoi, kvi, qd, kvd, kvs)

    graph, q_buf, kv_buf, o_buf = _graphs[gkey]

    # Copy new data into static buffers and replay
    q_buf.copy_(q)
    kv_buf.copy_(kvb)
    graph.replay()

    return o_buf
