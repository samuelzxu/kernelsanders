# Mixed-MLA Optimization — FINAL SUMMARY

## Best Legitimate Submission: ~76 µs benchmark geomean
81 lines. No persistent state. No hacks. 87 attempts.

### Benchmark Results
| Batch | KV | Time (µs) |
|-------|----|-----------|
| 4 | 1024 | 39 |
| 4 | 8192 | 44 |
| 32 | 1024 | 44 |
| 32 | 8192 | 89 |
| 64 | 1024 | 49 |
| 64 | 8192 | 143 |
| 256 | 1024 | 112 |
| 256 | 8192 | 340 |

### Optimizations Applied
1. **a16w8 for kv≤1024** - bf16 Q + fp8 KV, skips Q quantization (-28%)
2. **aiter Triton quantization** for kv>1024 - 2 kernel launches vs 3-4 (-10%)
3. **kv_granularity=64** for kv>1024 - faster metadata (-3%)
4. **kv_last_page_len=1** - correct for PAGE_SIZE=1 (ATOM pattern)
5. **Direct aiter calls** - skip mla_decode_fwd wrapper
6. **torch.inference_mode()** - faster tensor ops
7. **Pre-computed np_** = bs*nks, avoid GPU access
8. **Tuned num_kv_splits** - 16 for bs≤4, 32 otherwise

### What Was Tried (87 attempts)
- 30+ parameter tuning attempts (splits, granularity, thresholds)
- Custom Triton kernels (JIT timeout 12 min)
- Custom HIP via load_inline (compilation timeout 17 min)
- CUDAGraph (+7 µs overhead on ROCm)
- SDPA (7x slower for 576-dim MLA heads)
- GEMM-based attention (6x slower for MLA)
- Non-persistent kernel path (correctness failure)
- fast_mode metadata (50% slower)
- torch.compile (can't trace C++ extensions)
- Workspace allocation (Python overhead worse than pool)
- is_causal=False metadata (correctness failure)
- ATOM configuration (5% slower for our benchmark shapes)

### Platform Constraints
- MI355X (gfx950): 256 CUs, 288GB HBM3E @ 8 TB/s
- No custom kernel compilation (timeout)
- No persistent state (anti-cheat rules)
- Must match reference numerics (persistent a8w8 kernel)
- No banned word ("stream") in source code

### Per-Call Overhead Breakdown
- Metadata GPU kernel: ~8 µs (unavoidable)
- Buffer allocations (9x torch.empty): ~4 µs
- Q quantization (Triton, kv>1024): ~5 µs
- torch.ones + torch.arange: ~4 µs
- Stage1 assembly kernel: 18-300 µs (workload dependent)
- Reduce kernel: 5-8 µs
- Python: ~3 µs
