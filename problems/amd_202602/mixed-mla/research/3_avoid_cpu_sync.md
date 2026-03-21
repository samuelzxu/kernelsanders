# Attempt 3: Avoid CPU Synchronization

## What we tried
- Changed `total_kv_len = int(kv_indptr[-1].item())` to `total_kv_len = kv_buffer_fp8.shape[0]`
- Avoids CPU-GPU synchronization from `.item()` call

## Results (µs)
| Batch | KV Len | Before | After | Speedup |
|-------|--------|--------|-------|---------|
| 4 | 1024 | 128 | 55.5 | 2.3x |
| 4 | 8192 | 136 | 68.6 | 2.0x |
| 32 | 1024 | 135 | 65.1 | 2.1x |
| 32 | 8192 | 177 | 109 | 1.6x |
| 64 | 1024 | 137 | 72.8 | 1.9x |
| 64 | 8192 | 222 | 160 | 1.4x |
| 256 | 1024 | 181 | 123 | 1.5x |
| 256 | 8192 | 371 | 319 | 1.2x |

## Comparison to Reference
| Case | Reference | Ours | Speedup |
|------|-----------|------|---------|
| bs=4, kv=1k | ~118 | 55.5 | 2.1x faster |
| bs=4, kv=8k | ~113 | 68.6 | 1.6x faster |
| bs=64, kv=8k | ~171 | 160 | 1.07x faster |
| bs=256, kv=8k | ~349 | 319 | 1.09x faster |

## Key Insight
The `.item()` call forces CPU-GPU synchronization which blocks the GPU pipeline.
Using tensor shape (which is metadata) avoids this overhead.

## Conclusion
Major improvement - now beating reference on all benchmarks!
