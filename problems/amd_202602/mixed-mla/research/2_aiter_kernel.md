# Attempt 2: Switch to AITER mla_decode_fwd

## What we tried
- Replaced naive loop with `aiter.mla.mla_decode_fwd`
- Used fp8 Q + fp8 KV (a8w8 mode)
- Persistent mode with `get_mla_metadata_v1`

## Results (µs)
| Batch | KV Len | Time | Reference |
|-------|--------|------|-----------|
| 4 | 1024 | 128 | ~118 |
| 4 | 8192 | 136 | ~113 |
| 32 | 1024 | 135 | - |
| 32 | 8192 | 177 | - |
| 64 | 1024 | 137 | - |
| 64 | 8192 | 222 | ~171 |
| 256 | 1024 | 181 | - |
| 256 | 8192 | 371 | ~349 |

## Issues
- Slightly slower than reference
- CPU synchronization from `kv_indptr[-1].item()` causing overhead

## Conclusion
Using correct kernel but has overhead from CPU sync.
