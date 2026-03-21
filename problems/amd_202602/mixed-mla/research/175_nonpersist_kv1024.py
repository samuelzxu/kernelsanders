"""
MLA decode - non-persistent mla_decode_fwd for kv<=1024 assembly path.

Key insight: non-persistent mode skips ALL metadata overhead:
- No get_mla_metadata_info_v1 call (~1µs)
- No 6 metadata buffer allocations (~2µs)
- No get_mla_metadata_v1 GPU kernel (~3µs)
Total savings: ~6µs per kv<=1024 assembly call

Trade-off: uses Triton stage2 reduce kernel (JIT compiled during warmup).
For kv<=1024 with nhead=16, the function auto-selects optimal nks.

kv>1024 path stays on persistent mode (proven optimal with kvg=32, nks=32).
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

from aiter.mla import mla_decode_fwd
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

_KVI_SHORT = torch.arange(256 * 1024, dtype=torch.int32, device="cuda")


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

    # Non-persistent mode for kv<=1024: skip ALL metadata overhead
    if kv_seq_len <= 1024:
        kvi = _KVI_SHORT[:n]
        klp = qo_indptr[1:bs+1]
        o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")
        kv_buffer_4d = kv_buffer.view(n, 1, NKV, DQ)

        # Non-persistent mode: no metadata needed, auto nks selection
        mla_decode_fwd(
            q.view(-1, NH, DQ),
            kv_buffer_4d,
            o,
            qo_indptr, kv_indptr, kvi, klp,
            1,  # max_seqlen_q
            page_size=1,
            nhead_kv=NKV,
            sm_scale=SM_SCALE,
            logit_cap=0.0,
            num_kv_splits=8,
            q_scale=None,
            kv_scale=kv_scale,
            intra_batch_mode=False,
        )
        return o

    # Persistent mode for kv>1024 (proven optimal)
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
    wm, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
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
    lg = torch.empty((np_, 1, NH, DV), dtype=FP32, device="cuda")
    ls = torch.empty((np_, 1, NH, 1), dtype=FP32, device="cuda")
    o = torch.empty((bs, NH, DV), dtype=BF16, device="cuda")

    aiter.mla_decode_stage1_asm_fwd(
        q_input.view(-1, NH, DQ),
        kv_buffer.view(n, 1, NKV, DQ),
        qo_indptr, kv_indptr, kvi, klp,
        None, wm, wi, wis, 1, 1, NKV, SM_SCALE,
        lg, ls, o, q_scale, kv_scale)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o
