# Attempt 16: KV-length based split tuning

## Changes
- bs <= 4: 16 splits
- bs 5-64: 32 for kv<=4096, 48 for kv>4096
- bs > 64: 48 for kv<=4096, 64 for kv>4096

## Hypothesis
Long KV sequences (kv=8192) might benefit from more splits to
better utilize memory bandwidth across split computations.

## Previous Best (Attempt 15)
| Batch | KV Len | Splits | Time (us) |
|-------|--------|--------|-----------|
| 4 | 1024 | 16 | 56.7 |
| 4 | 8192 | 16 | 65.4 |
| 32 | 1024 | 32 | 64.4 |
| 32 | 8192 | 32 | 108 |
| 64 | 1024 | 32 | 73.6 |
| 64 | 8192 | 32 | 154 |
| 256 | 1024 | 64 | 123 |
| 256 | 8192 | 64 | 320 |

## New Configuration
| Batch | KV Len | Splits |
|-------|--------|--------|
| 4 | 1024 | 16 |
| 4 | 8192 | 16 |
| 32 | 1024 | 32 |
| 32 | 8192 | 48 |
| 64 | 1024 | 32 |
| 64 | 8192 | 48 |
| 256 | 1024 | 48 |
| 256 | 8192 | 64 |

## Results - REGRESSION
| Batch | KV Len | Previous | New | Change |
|-------|--------|----------|-----|--------|
| 4 | 1024 | 56.7 | 56.1 | -1.1% ✓ |
| 4 | 8192 | 65.4 | 64.3 | -1.7% ✓ |
| 32 | 1024 | 64.4 | 64.6 | +0.3% |
| 32 | 8192 | 108 | 115 | +6.5% ❌ |
| 64 | 1024 | 73.6 | 74.5 | +1.2% |
| 64 | 8192 | 154 | 168 | +9.1% ❌ |
| 256 | 1024 | 123 | 130 | +5.7% ❌ |
| 256 | 8192 | 320 | 330 | +3.1% ❌ |

## Analysis
- Small batches improved slightly with same splits (variance)
- Medium/large batches regressed significantly with 48 splits
- 48 splits seems to be a bad choice for these workloads
- Reverted to simple configuration (16 for bs<=4, 32 otherwise)
