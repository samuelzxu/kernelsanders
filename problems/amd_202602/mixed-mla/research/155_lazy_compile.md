# Attempt 155: Lazy torch.compile to avoid assembly interference

## Issue with 153
torch.compile at module level caused +2-5µs regression on assembly configs.
Hypothesis: module-level compilation triggers Triton/Inductor backend loading
that interferes with aiter assembly initialization.

## Fix
Lazy compilation: torch.compile called on first GEMM use (during warmup),
not at module load time. This ensures aiter modules load first without
interference from Inductor/Triton backend.

## Expected
- GEMM configs: same improvement as 153 (21.6, 39.2µs for kv=1024)
- Assembly configs: no regression (same as 137: 88.6, 141, 113, 338µs)
- bs=4/kv=8192: 42.5µs (uncompiled, same as 137)
