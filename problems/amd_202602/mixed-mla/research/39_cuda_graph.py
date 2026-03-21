"""
MLA decode kernel — CUDA graph approach.
Capture the kernel dispatch as a graph and replay it.
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

# Caches
_cache: dict[tuple, tuple] = {}
_kvi: dict[int, torch.Tensor] = {}
_obuf: dict[int, torch.Tensor] = {}
_graphs: dict[tuple, tuple] = {}  # (bs,) -> (graph, q_buf, kv_buf, o_buf, ...)

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
    np_ = rpm.size(0)
    lg = torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda")
    ls = torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, klp, lg, ls)


def _build_graph(bs, q, kvb, qoi, kvi, qd, kvd, kvs):
    """Capture a CUDA graph for the kernel execution."""
    nks = 16 if bs <= 4 else 32
    n = kvb.shape[0]
    tq = q.shape[0]

    # Allocate static buffers for graph capture
    q_static = q.clone()
    kv_static = kvb.clone()
    o_static = torch.empty((tq, NH, DV), dtype=BF16, device="cuda")

    key = (bs, nks, qd, kvd)
    if key not in _cache:
        _cache[key] = _build(bs, nks, qd, kvd, qoi, kvi)
    wm, wi, wis, ri, rfm, rpm, klp, lg, ls = _cache[key]

    if n not in _kvi:
        _kvi[n] = torch.arange(n, dtype=torch.int32, device="cuda")

    # Warmup run
    q_view = q_static.view(-1, NH, DQ)
    kv_view = kv_static.view(n, PAGE_SIZE, NKV, DQ)
    qs = None
    _stage1(q_view, kv_view, qoi, kvi, _kvi[n], klp,
            None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o_static, qs, kvs)
    _reduce(lg, ls, ri, rfm, rpm, 1, o_static, None)
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _stage1(q_view, kv_view, qoi, kvi, _kvi[n], klp,
                None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE, lg, ls, o_static, qs, kvs)
        _reduce(lg, ls, ri, rfm, rpm, 1, o_static, None)

    return graph, q_static, kv_static, o_static


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qoi, kvi, cfg = data
    bs = cfg["batch_size"]

    if bs <= 32:
        kvb = kv_data["bf16"]
        qd, kvd = BF16, BF16
        kvs = None
    else:
        kvb, kvs = kv_data["fp8"]
        qd, kvd = BF16, FP8_DTYPE

    graph_key = (bs, qd, kvd)

    if graph_key not in _graphs:
        # First call: build and capture graph
        _graphs[graph_key] = _build_graph(bs, q, kvb, qoi, kvi, qd, kvd, kvs)

    graph, q_static, kv_static, o_static = _graphs[graph_key]

    # Copy new data into static buffers
    q_static.copy_(q)
    kv_static.copy_(kvb)

    # Replay graph
    graph.replay()

    return o_static
