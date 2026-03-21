"""
MLA decode - honest, no persistent state.
a16w8 for kv<=1024 (no quant), a8w8+Triton quant for kv>1024.
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.triton.quant.quant import dynamic_per_tensor_quant_fp8_i8

FP8 = aiter_dtypes.fp8
BF16 = torch.bfloat16
F32 = torch.float32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM = 1.0 / (DQ ** 0.5)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qoi, kvi, cfg = data
    bs = cfg["batch_size"]
    kvl = cfg.get("kv_seq_len", 1024)
    kvb, kvs = kv_data["fp8"]
    n = kvb.shape[0]
    nks = 16 if bs <= 4 else 32

    if kvl <= 1024:
        qi, qs, qd = q, None, BF16
    else:
        qi = torch.empty(q.shape[0], q.shape[1], q.shape[2], dtype=FP8, device="cuda")
        qs = torch.zeros(1, dtype=F32, device="cuda")
        dynamic_per_tensor_quant_fp8_i8(qi.view(-1, DQ), q.view(-1, DQ), qs)
        qd = FP8

    kvd = kvb.dtype
    info = get_mla_metadata_info_v1(bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
                                     num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    klp = (kvi[1:] - kvi[:-1]).to(torch.int32)
    idx = torch.arange(n, dtype=torch.int32, device="cuda")

    get_mla_metadata_v1(qoi, kvi, klp, NH, NKV, True, wm, wis, wi, ri, rfm, rpm,
                        page_size=1, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1,
                        fast_mode=False, max_split_per_batch=nks,
                        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)

    np_ = rpm.size(0)
    lg = torch.empty((np_, 1, NH, DV), dtype=F32, device="cuda")
    ls = torch.empty((np_, 1, NH, 1), dtype=F32, device="cuda")
    o = torch.empty((q.shape[0], NH, DV), dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        qi.view(-1, NH, DQ), kvb.view(n, 1, NKV, DQ),
        qoi, kvi, idx, klp, None, wm, wi, wis, 1, 1, NKV, SM,
        lg, ls, o, qs, kvs)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
