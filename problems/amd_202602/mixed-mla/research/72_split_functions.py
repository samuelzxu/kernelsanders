"""
MLA decode - honest, torch.compile for overhead reduction.
a16w8 for kv<=1024, a8w8+Triton quant for kv>1024.
"""

import torch
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_v1
from aiter.ops.triton.quant.quant import dynamic_per_tensor_quant_fp8_i8

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
I32 = torch.int32
U64 = torch.uint64
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)
CU_NUM = torch.cuda.get_device_properties(0).multi_processor_count


def quantize_fp8_triton(tensor):
    t2d = tensor.reshape(-1, tensor.shape[-1])
    qx = torch.empty_like(t2d, dtype=FP8_DTYPE)
    scale = torch.zeros(1, dtype=FP32, device=tensor.device)
    dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
    return qx.reshape(tensor.shape), scale


def _run_a16w8(q, kv_buffer, kv_scale, qo_indptr, kv_indptr, bs, nks):
    """bf16 Q + fp8 KV path (no quantization)."""
    n = kv_buffer.shape[0]
    tc = bs
    wm = torch.empty(2, dtype=U64, device="cuda")
    wi = torch.empty(CU_NUM + 1, dtype=I32, device="cuda")
    wis = torch.empty(tc * nks, 8, dtype=I32, device="cuda")
    ri = torch.empty(tc + 1, dtype=I32, device="cuda")
    rfm = torch.empty(tc, 2, dtype=I32, device="cuda")
    rpm = torch.empty(tc * nks, dtype=I32, device="cuda")
    klp = (kv_indptr[1:] - kv_indptr[:-1]).to(I32)
    kvi = torch.arange(n, dtype=I32, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=16,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=BF16, dtype_kv=kv_buffer.dtype)

    np_ = rpm.size(0)
    lg = torch.empty(np_, 1, NH, DV, dtype=FP32, device="cuda")
    ls = torch.empty(np_, 1, NH, 1, dtype=FP32, device="cuda")
    o = torch.empty(q.shape[0], NH, DV, dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q.reshape(-1, NH, DQ), kv_buffer.reshape(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE, lg, ls, o, None, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o


def _run_a8w8(q, kv_buffer, kv_scale, qo_indptr, kv_indptr, bs, nks):
    """fp8 Q + fp8 KV path (Triton quantization)."""
    q_fp8, q_scale = quantize_fp8_triton(q)
    n = kv_buffer.shape[0]
    tc = bs
    wm = torch.empty(2, dtype=U64, device="cuda")
    wi = torch.empty(CU_NUM + 1, dtype=I32, device="cuda")
    wis = torch.empty(tc * nks, 8, dtype=I32, device="cuda")
    ri = torch.empty(tc + 1, dtype=I32, device="cuda")
    rfm = torch.empty(tc, 2, dtype=I32, device="cuda")
    rpm = torch.empty(tc * nks, dtype=I32, device="cuda")
    klp = (kv_indptr[1:] - kv_indptr[:-1]).to(I32)
    kvi = torch.arange(n, dtype=I32, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=16,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=FP8_DTYPE, dtype_kv=kv_buffer.dtype)

    np_ = rpm.size(0)
    lg = torch.empty(np_, 1, NH, DV, dtype=FP32, device="cuda")
    ls = torch.empty(np_, 1, NH, 1, dtype=FP32, device="cuda")
    o = torch.empty(q.shape[0], NH, DV, dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q_fp8.reshape(-1, NH, DQ), kv_buffer.reshape(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE, lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)
    kv_buffer, kv_scale = kv_data["fp8"]
    nks = 16 if bs <= 4 else 32

    if kv_seq_len <= 1024:
        return _run_a16w8(q, kv_buffer, kv_scale, qo_indptr, kv_indptr, bs, nks)
    else:
        return _run_a8w8(q, kv_buffer, kv_scale, qo_indptr, kv_indptr, bs, nks)
