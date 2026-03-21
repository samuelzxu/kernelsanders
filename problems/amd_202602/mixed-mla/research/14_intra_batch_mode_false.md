# Attempt 14: intra_batch_mode=False

## Changes
- Changed `intra_batch_mode=True` to `intra_batch_mode=False` in:
  - `get_mla_metadata_info_v1()`
  - `get_mla_metadata_v1()`
  - `mla_decode_fwd()`

## Hypothesis
`intra_batch_mode` controls how work is distributed across batches.
- True: Work items can span multiple queries within a batch
- False: Each query is processed independently

For decode (q_seq_len=1), intra_batch_mode=False might reduce coordination overhead.

## Previous Best (Attempt 10)
| Batch | KV Len | Time (us) |
|-------|--------|-----------|
| 4 | 1024 | 56.6 |
| 4 | 8192 | 65.8 |
| 32 | 1024 | 64.0 |
| 32 | 8192 | 108 |
| 64 | 1024 | 73.2 |
| 64 | 8192 | 156 |
| 256 | 1024 | 122 |
| 256 | 8192 | 323 |

## Results - MASSIVE REGRESSION
| Batch | KV Len | Previous | New | Change |
|-------|--------|----------|-----|--------|
| 4 | 1024 | 56.6 | 124 | +119% ❌ |
| 4 | 8192 | 65.8 | 525 | +698% ❌ |
| 32 | 1024 | 64.0 | 555 | +767% ❌ |
| 32 | 8192 | 108 | 600 | +456% ❌ |
| 64 | 1024 | 73.2 | 606 | +728% ❌ |
| 64 | 8192 | 156 | 688 | +341% ❌ |
| 256 | 1024 | 122 | 883 | +624% ❌ |
| 256 | 8192 | 323 | 1076 | +233% ❌ |

## Analysis
`intra_batch_mode=False` is CATASTROPHICALLY slower (2-10x worse).
The intra_batch_mode=True is essential for batched decode.

## Conclusion
MUST keep `intra_batch_mode=True`. Reverted.
