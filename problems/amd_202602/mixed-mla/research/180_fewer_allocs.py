"""
MLA decode - reduce allocation count in assembly path.

Current assembly path has 10+ torch.empty calls per invocation:
- 6 metadata buffers (wm, wi, wis, ri, rfm, rpm)
- lg, ls, o = 3 more
- kvi = 1 (torch.arange)
- Q quant: qx, scale = 2 more (for kv>1024)
Total: 12-14 allocations per call!

Each allocation calls into HIP memory pool (~0.3µs per call).
Reducing from 14 to fewer could save 1-3µs.

Strategy: unroll the metadata list comprehension into explicit calls,
and merge lg+ls into one allocation (proven safe in attempt 147).
"""

import os
os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '1')
os.environ.setdefault('PYTORCH_TUNABLEOP_VERBOSE', '0')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS', '100')
os.environ.setdefault('PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS', '50')

import torch
import torch.nn.functional as F
from task import input_t, output_t

import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.ops.triton.quant.quant import dynamic_per_tensor_quant_fp8_i8

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
NH = 16
NKV = 1
DQ = 576
DV = 512
SM_SCALE = 1.0 / (DQ ** 0.5)


def _gemm_attn_self_alloc(q_3d, kv_t, v):
    bs = q_3d.shape[0]
    kv_len = kv_t.shape[2]
    scores = torch.empty(bs, NH, kv_len, dtype=BF16, device=q_3d.device)
    torch.baddbmm(scores, q_3d, kv_t, beta=0, alpha=SM_SCALE, out=scores)
    return torch.bmm(F.softmax(scores, dim=-1, dtype=FP32).to(BF16), v)


_compiled_gemm_short = torch.compile(_gemm_attn_self_alloc)
_compiled_gemm_long = torch.compile(_gemm_attn_self_alloc)


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):
        kv = kv_data["bf16"].view(bs, kv_seq_len, DQ)
        q_3d = q.view(bs, NH, DQ)
        v = kv[:, :, :DV]
        kv_t = kv.transpose(-2, -1)

        if kv_seq_len <= 1024:
            return _compiled_gemm_short(q_3d, kv_t, v)
        else:
            return _compiled_gemm_long(q_3d, kv_t, v)

    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]

    if kv_seq_len <= 1024:
        q_input, q_scale, qd = q, None, BF16
        kvg, nks = 32, 8
    else:
        t2d = q.view(-1, DQ)
        qx = torch.empty_like(t2d, dtype=FP8_DTYPE)
        scale = torch.zeros(1, dtype=FP32, device=q.device)
        dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
        q_input, q_scale, qd = qx.view(q.shape), scale, FP8_DTYPE
        kvg, nks = 32, 32

    kvd = kv_buffer.dtype
    info = get_mla_metadata_info_v1(
        bs, 1, NH, qd, kvd, is_sparse=False, fast_mode=False,
        num_kv_splits=nks, intra_batch_mode=True)

    # Unrolled metadata allocation (avoids list comprehension overhead)
    wm = torch.empty(info[0][0], dtype=info[0][1], device="cuda")
    wi = torch.empty(info[1][0], dtype=info[1][1], device="cuda")
    wis = torch.empty(info[2][0], dtype=info[2][1], device="cuda")
    ri = torch.empty(info[3][0], dtype=info[3][1], device="cuda")
    rfm = torch.empty(info[4][0], dtype=info[4][1], device="cuda")
    rpm = torch.empty(info[5][0], dtype=info[5][1], device="cuda")

    klp = qo_indptr[1:bs+1]
    kvi = torch.arange(n, dtype=torch.int32, device="cuda")

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, klp, NH // NKV, NKV, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=1, kv_granularity=kvg,
        max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=nks,
        intra_batch_mode=True, dtype_q=qd, dtype_kv=kvd)

    np_ = bs * nks

    # Merge lg+ls into single allocation (saves 1 alloc call)
    lgls_total = np_ * NH * (DV + 1)
    lgls_buf = torch.empty(lgls_total, dtype=FP32, device="cuda")
    lg_count = np_ * NH * DV
    lg = lgls_buf[:lg_count].view(np_, 1, NH, DV)
    ls = lgls_buf[lg_count:].view(np_, 1, NH, 1)

    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer.view(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE,
        lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
