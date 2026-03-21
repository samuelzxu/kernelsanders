"""
MLA decode - honest, tuned kv_granularity.
a16w8 for kv<=1024, a8w8+Triton quant for kv>1024.
Workspace allocation: single i32 + single f32 alloc.
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
U64 = torch.uint64
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)
CU = torch.cuda.get_device_properties(0).multi_processor_count


def quantize_fp8_triton(tensor):
    t2d = tensor.view(-1, tensor.shape[-1])
    qx = torch.empty_like(t2d, dtype=FP8_DTYPE)
    scale = torch.zeros(1, dtype=FP32, device=tensor.device)
    dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
    return qx.view(tensor.shape), scale


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]
    nks = 16 if bs <= 4 else 32
    tc = bs

    if kv_seq_len <= 1024:
        q_input, q_scale, qd = q, None, BF16
        kvg = 16
    else:
        q_input, q_scale = quantize_fp8_triton(q)
        qd = FP8_DTYPE
        kvg = 64

    kvd = kv_buffer.dtype

    # Compute sizes
    wis_sz = tc * nks * 8
    rpm_sz = tc * nks
    wi_sz = CU + 1
    ri_sz = tc + 1
    rfm_sz = tc * 2

    # Single i32 workspace for wis + rpm + wi + ri + rfm
    i32_ws = torch.empty(wis_sz + rpm_sz + wi_sz + ri_sz + rfm_sz, dtype=I32, device="cuda")
    p = 0
    wis = i32_ws[p:p+wis_sz].view(tc * nks, 8); p += wis_sz
    rpm = i32_ws[p:p+rpm_sz]; p += rpm_sz
    wi = i32_ws[p:p+wi_sz]; p += wi_sz
    ri = i32_ws[p:p+ri_sz]; p += ri_sz
    rfm = i32_ws[p:p+rfm_sz].view(tc, 2)

    wm = torch.empty(2, dtype=U64, device="cuda")
    klp = (kv_indptr[1:] - kv_indptr[:-1]).to(I32)
    kvi = torch.arange(n, dtype=I32, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH // NKV, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=kvg,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)

    np_ = rpm.size(0)
    # Single f32 workspace for lg + ls
    f32_total = np_ * NH * DV + np_ * NH
    f32_ws = torch.empty(f32_total, dtype=FP32, device="cuda")
    lg = f32_ws[:np_ * NH * DV].view(np_, 1, NH, DV)
    ls = f32_ws[np_ * NH * DV:].view(np_, 1, NH, 1)

    o = torch.empty(q.shape[0], NH, DV, dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer.view(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE,
        lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
