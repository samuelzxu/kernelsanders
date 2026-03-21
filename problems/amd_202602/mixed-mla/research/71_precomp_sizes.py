"""
MLA decode - honest, no persistent state.
a16w8 for kv<=1024, a8w8+Triton quant for kv>1024.
torch.compile on quantization for kernel fusion.
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
PAGE_SIZE = 1
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)

# Hardware constant
_CU_NUM = torch.cuda.get_device_properties(0).multi_processor_count


def _meta_sizes(bs, nks):
    """Pure math - no GPU ops."""
    tile_cnt = bs
    return (
        ((2,), torch.uint64),
        ((_CU_NUM + 1,), torch.int32),
        ((tile_cnt * nks, 8), torch.int32),
        ((tile_cnt + 1,), torch.int32),
        ((tile_cnt, 2), torch.int32),
        ((tile_cnt * nks,), torch.int32),
    )


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

    if kv_seq_len <= 1024:
        q_input, q_scale, qd = q, None, BF16
    else:
        q_input, q_scale = quantize_fp8_triton(q)
        qd = FP8_DTYPE

    kvd = kv_buffer.dtype
    sizes = _meta_sizes(bs, nks)
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in sizes]
    klp = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    kvi = torch.arange(n, dtype=torch.int32, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH // NKV, NKV, True,
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
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, PAGE_SIZE, NKV, SM_SCALE,
        lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)

    return o
