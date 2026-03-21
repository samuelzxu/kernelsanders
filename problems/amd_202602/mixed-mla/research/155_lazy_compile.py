"""
MLA decode - lazy torch.compile for kv<=1024 GEMM only.

Same as 153 but with lazy compilation to avoid module-load overhead
that might interfere with assembly path initialization.

torch.compile is called on first use (during warmup), not at module load.
"""

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

_compiled_fn = None


def _gemm_attn_short(q_3d, kv_t, v, scores_buf):
    torch.baddbmm(scores_buf, q_3d, kv_t, beta=0, alpha=SM_SCALE, out=scores_buf)
    return torch.bmm(F.softmax(scores_buf, dim=-1, dtype=FP32).to(BF16), v)


def quantize_fp8_triton(tensor):
    t2d = tensor.view(-1, tensor.shape[-1])
    qx = torch.empty_like(t2d, dtype=FP8_DTYPE)
    scale = torch.zeros(1, dtype=FP32, device=tensor.device)
    dynamic_per_tensor_quant_fp8_i8(qx, t2d, scale)
    return qx.view(tensor.shape), scale


@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _compiled_fn
    q, kv_data, qo_indptr, kv_indptr, cfg = data
    bs = cfg["batch_size"]
    kv_seq_len = cfg.get("kv_seq_len", 1024)

    # GEMM path for small cases
    if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):
        kv = kv_data["bf16"].view(bs, kv_seq_len, DQ)
        q_3d = q.view(bs, NH, DQ)
        v = kv[:, :, :DV]
        kv_t = kv.transpose(-2, -1)

        if kv_seq_len <= 1024:
            # Lazy compile: deferred to first call to avoid import overhead
            if _compiled_fn is None:
                _compiled_fn = torch.compile(_gemm_attn_short, dynamic=True)
            scores_buf = torch.empty(bs, NH, kv_seq_len, dtype=BF16, device=q.device)
            return _compiled_fn(q_3d, kv_t, v, scores_buf)
        else:
            # Original path for kv>1024 (torch.compile hurts large N)
            s = torch.baddbmm(
                torch.empty(bs, NH, kv_seq_len, dtype=BF16, device=q.device),
                q_3d, kv_t, beta=0, alpha=SM_SCALE)
            return torch.bmm(F.softmax(s, dim=-1, dtype=FP32).to(BF16), v)

    kv_buffer, kv_scale = kv_data["fp8"]
    n = kv_buffer.shape[0]

    if kv_seq_len <= 1024:
        q_input, q_scale, qd = q, None, BF16
        kvg, nks = 16, 8
    else:
        q_input, q_scale = quantize_fp8_triton(q)
        qd, kvg = FP8_DTYPE, 64
        nks = 32

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
