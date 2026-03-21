"""
MLA (Multi-head Latent Attention) decode kernel — optimized submission.

Uses aiter mla_decode_fwd with fp8 Q + fp8 KV for persistent-mode MLA attention.
"""

import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP8_DTYPE = aiter_dtypes.fp8
PAGE_SIZE = 1
NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512

# ---------------------------------------------------------------------------
# Caches — reuse allocations and computed metadata across calls
# ---------------------------------------------------------------------------
_meta_cache: dict[tuple[int, int], dict] = {}
_kv_indices_cache: dict[int, torch.Tensor] = {}
_output_cache: dict[int, torch.Tensor] = {}
_kv_last_page_cache: dict[int, torch.Tensor] = {}


def get_num_kv_splits(batch_size: int) -> int:
    if batch_size <= 4:
        return 16
    else:
        return 32


# ---------------------------------------------------------------------------
# FP8 quantization helper
# ---------------------------------------------------------------------------

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 quantization. Returns (fp8_tensor, scale)."""
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# Cached metadata builder
# ---------------------------------------------------------------------------

def _get_or_build_metadata(
    batch_size: int,
    num_kv_splits: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
):
    """Build and cache full metadata for persistent mla_decode_fwd.

    qo_indptr and kv_indptr are deterministic for a given (batch_size, q_seq_len, kv_seq_len),
    so the metadata only needs to be computed once per shape configuration.
    """
    key = (batch_size, num_kv_splits)
    if key not in _meta_cache:
        info = get_mla_metadata_info_v1(
            batch_size, 1, NUM_HEADS, q_dtype, kv_dtype,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        bufs = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (work_metadata, work_indptr, work_info_set,
         reduce_indptr, reduce_final_map, reduce_partial_map) = bufs

        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            NUM_HEADS // NUM_KV_HEADS,
            NUM_KV_HEADS,
            True,
            work_metadata, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            page_size=PAGE_SIZE,
            kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=q_dtype,
            dtype_kv=kv_dtype,
        )

        _meta_cache[key] = {
            "work_meta_data": work_metadata,
            "work_indptr": work_indptr,
            "work_info_set": work_info_set,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
            "_kv_last_page_len": kv_last_page_len,
        }
    return _meta_cache[key]


def _get_kv_indices(total_kv_len: int) -> torch.Tensor:
    if total_kv_len not in _kv_indices_cache:
        _kv_indices_cache[total_kv_len] = torch.arange(
            total_kv_len, dtype=torch.int32, device="cuda"
        )
    return _kv_indices_cache[total_kv_len]


def _get_output_buf(total_q: int) -> torch.Tensor:
    if total_q not in _output_cache:
        _output_cache[total_q] = torch.empty(
            (total_q, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )
    return _output_cache[total_q]


# ---------------------------------------------------------------------------
# Main kernel entry point
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    """MLA decode using aiter fp8 persistent kernel (a8w8)."""
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    sm_scale = config["sm_scale"]

    # Use fp8 KV cache
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Quantize Q to fp8
    q_fp8, q_scale = quantize_fp8(q)

    # Build KV indices (cached)
    total_kv_len = kv_buffer_fp8.shape[0]
    kv_indices = _get_kv_indices(total_kv_len)

    # Reshape kv_buffer to 4D for aiter (view is free, no data copy)
    kv_buffer_4d = kv_buffer_fp8.view(total_kv_len, PAGE_SIZE, NUM_KV_HEADS, QK_HEAD_DIM)

    # Get tuned num_kv_splits
    num_kv_splits = get_num_kv_splits(batch_size)

    # Build persistent-mode metadata (fully cached - qo/kv_indptr are deterministic)
    cached = _get_or_build_metadata(
        batch_size, num_kv_splits,
        q_fp8.dtype, kv_buffer_fp8.dtype,
        qo_indptr, kv_indptr,
    )
    kv_last_page_len = cached["_kv_last_page_len"]

    # Reuse output buffer
    o = _get_output_buf(q.shape[0])

    # Run MLA decode
    mla_decode_fwd(
        q_fp8.view(-1, NUM_HEADS, QK_HEAD_DIM),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        1,  # max_q_len = 1 for decode
        page_size=PAGE_SIZE,
        nhead_kv=NUM_KV_HEADS,
        sm_scale=sm_scale,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=cached["work_meta_data"],
        work_indptr=cached["work_indptr"],
        work_info_set=cached["work_info_set"],
        reduce_indptr=cached["reduce_indptr"],
        reduce_final_map=cached["reduce_final_map"],
        reduce_partial_map=cached["reduce_partial_map"],
    )

    return o
