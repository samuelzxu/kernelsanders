"""
MLA decode - honest, no persistent state.
Uses mla_decode_fwd (non-persistent) to skip metadata computation.
a8w8 for correctness match with reference.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
PAGE_SIZE = 1
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)


def quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]

    # fp8 Q + fp8 KV (a8w8) matching reference
    kv_buffer, kv_scale = kv_data["fp8"]
    q_input, q_scale = quantize_fp8(q)

    n = kv_buffer.shape[0]
    kv_indices = torch.arange(n, dtype=torch.int32, device="cuda")
    kv_buffer_4d = kv_buffer.view(n, PAGE_SIZE, NKV, DQ)
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    o = torch.empty((q.shape[0], NH, DV), dtype=BF16, device="cuda")

    # Non-persistent: let mla_decode_fwd compute metadata internally
    # This skips the expensive get_mla_metadata_v1 GPU kernel
    mla_decode_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer_4d,
        o,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        1,
        page_size=PAGE_SIZE,
        nhead_kv=NKV,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=32,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
    )
    return o
