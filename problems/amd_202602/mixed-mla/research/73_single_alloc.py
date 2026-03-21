"""
MLA decode - honest, minimized allocations.
Single buffer allocation + slicing instead of 9 separate torch.empty calls.
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
CU = torch.cuda.get_device_properties(0).multi_processor_count


def quantize_fp8_triton(tensor):
    t2d = tensor.reshape(-1, tensor.shape[-1])
    qx = torch.empty_like(t2d, dtype=FP8_DTYPE)
    scale = torch.zeros(1, dtype=FP32, device=tensor.device)
    dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
    return qx.reshape(tensor.shape), scale


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)
    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]
    nks = 16 if bs <= 4 else 32
    tc = bs  # tile_count

    if kv_seq_len <= 1024:
        q_input, q_scale, qd = q, None, BF16
    else:
        q_input, q_scale = quantize_fp8_triton(q)
        qd = FP8_DTYPE

    # Single i32 allocation for all metadata + kvi + klp
    # wis: tc*nks*8, rpm: tc*nks, wi: CU+1, ri: tc+1, rfm: tc*2, klp: bs, kvi: n
    i32_total = tc * nks * 8 + tc * nks + (CU + 1) + (tc + 1) + tc * 2 + bs + n
    i32_buf = torch.empty(i32_total, dtype=I32, device="cuda")

    off = 0
    wis = i32_buf[off:off + tc * nks * 8].view(tc * nks, 8); off += tc * nks * 8
    rpm = i32_buf[off:off + tc * nks]; off += tc * nks
    wi = i32_buf[off:off + CU + 1]; off += CU + 1
    ri = i32_buf[off:off + tc + 1]; off += tc + 1
    rfm = i32_buf[off:off + tc * 2].view(tc, 2); off += tc * 2
    klp = i32_buf[off:off + bs]; off += bs
    kvi_buf = i32_buf[off:off + n]; off += n

    # Fill klp and kvi
    klp.copy_((kv_indptr[1:] - kv_indptr[:-1]).to(I32))
    torch.arange(n, dtype=I32, device="cuda", out=kvi_buf)

    wm = torch.empty(2, dtype=U64, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=16,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kv_buffer.dtype)

    np_ = rpm.size(0)
    # Single f32 allocation for lg and ls
    f32_total = np_ * NH * DV + np_ * NH
    f32_buf = torch.empty(f32_total, dtype=FP32, device="cuda")
    lg = f32_buf[:np_ * NH * DV].view(np_, 1, NH, DV)
    ls = f32_buf[np_ * NH * DV:].view(np_, 1, NH, 1)

    o = torch.empty(q.shape[0], NH, DV, dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q_input.reshape(-1, NH, DQ), kv_buffer.reshape(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi_buf, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE, lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
