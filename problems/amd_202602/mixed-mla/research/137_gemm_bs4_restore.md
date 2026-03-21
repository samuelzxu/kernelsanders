# Attempt 137: Restore GEMM for bs<=4 (all kv lengths)

## Problem
Attempt 132 dropped GEMM for bs<=4/kv>1024 by changing the condition from:
  `if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):`
to:
  `if bs <= 32 and kv_seq_len <= 1024:`

This made bs=4/kv=8192 go through assembly (46µs) instead of GEMM (~10-15µs).
Geomean regressed from ~64-67µs to ~80µs.

## Root Cause Analysis
For bs=4/kv=8192:
- GEMM reads: ~37.7MB bf16 KV (L2 cache hit on second read for V), actual ~10-15µs
- Assembly reads: ~18.9MB fp8 KV, but fixed overhead: Q quant (~3µs) + metadata (~3µs)
  + attention kernel (~5µs) + reduce (~3µs) = ~15-20µs total

GEMM wins because:
1. Lower fixed overhead (no Q quant, no metadata, no reduce step)
2. 37.7MB KV fits in L2 cache (64MB), so V reads hit cache on second bmm

## Changes vs 132
- Restore GEMM condition: `if bs <= 4 or (bs <= 32 and kv_seq_len <= 1024):`
- This makes bs=4/kv=8192 use GEMM path again
- Assembly nks unchanged: 8 (kv<=1024), 32 (kv>1024)

## Expected Impact
- bs=4/kv=8192: 46µs → ~10-15µs (big win)
- All other configs: unchanged
- Geomean: ~80µs → ~65-70µs (recovering previous best)
