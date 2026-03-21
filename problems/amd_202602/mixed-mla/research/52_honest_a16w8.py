"""
MLA decode kernel - honest implementation, no persistent state.
Key optimizations (all within-invocation):
- bf16 Q + bf16 KV (a16w16) for short sequences (no quant overhead)
- bf16 Q + fp8 KV (a16w8) for long sequences (skip Q quant, fp8 bandwidth)
- Persistent mode matching reference
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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    # Kernel selection: bf16 for short seq, a16w8 for long seq
    if kv_seq_len <= 1024:
        # Short sequences: bf16 Q + bf16 KV (no quantization at all)
        kv_buffer = kv_data["bf16"]
        q_input = q
        q_scale, kv_scale = None, None
        qd, kvd = BF16, BF16
    else:
        # Long sequences: bf16 Q + fp8 KV (skip Q quantization, fp8 bandwidth)
        kv_buffer, kv_scale = kv_data["fp8"]
        q_input = q  # keep Q in bf16!
        q_scale = None
        qd, kvd = BF16, FP8_DTYPE

    n = kv_buffer.shape[0]
    nks = 16 if bs <= 4 else 32

    # Build metadata fresh each call
    info = get_mla_metadata_info_v1(
        bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
        num_kv_splits=nks, intra_batch_mode=True)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]

    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    kv_indices = torch.arange(n, dtype=torch.int32, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        NH // NKV, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=PAGE_SIZE, kv_granularity=16,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)

    np_ = rpm.size(0)
    lg = torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda")
    ls = torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda")
    o = torch.empty((q.shape[0], NH, DV), dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer.view(n, PAGE_SIZE, NKV, DQ),
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        None, wm, wi, wis,
        1, PAGE_SIZE, NKV, SM_SCALE,
        lg, ls, o, q_scale, kv_scale)

    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)

    return o
