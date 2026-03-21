# Diagnostic: Cost Breakdown (non-recheck benchmark mode)

## Results
| Component | bs=4 | bs=32 | bs=256 |
|-----------|------|-------|--------|
| No-op (Python) | 4.3 µs | 4.3 µs | 4.3 µs |
| Stage1 only | 22.6 µs | 25.2 µs | 85.5 µs |
| Full (s1+reduce) | 30.1 µs | 33.8 µs | 91.2 µs |
| **Reduce cost** | **7.5 µs** | **8.6 µs** | **5.7 µs** |
| **Stage1 kernel** | **18.3 µs** | **20.9 µs** | **81.2 µs** |
| **Python overhead** | **4.3 µs** | **4.3 µs** | **4.3 µs** |

## Analysis
- Python overhead is 4.3 µs constant - very low
- Stage1 assembly kernel dominates (60-89% of total)
- Reduce is ~6-8 µs, relatively constant
- For bs=4: reduce is 25% of total (7.5 / 30.1)
- For bs=256: reduce is only 6% of total (5.7 / 91.2)

## Optimization implications
- Eliminating reduce for bs=4 would save ~7.5 µs (25% improvement!)
- Further Python optimization is futile (4.3 µs is near zero)
- Stage1 kernel for bs=256 at 81 µs is the main bottleneck
- MXFP4 would reduce stage1 bandwidth by ~2x for large batches
