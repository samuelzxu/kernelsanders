# #139 BF16 Split-K Partials

## Approach
Patch `_USE_GEMM_SPLITK_BF16 = True` in gemm_afp4wfp4 module globals.
This changes split-K behavior:
- Partial results stored as bf16 (not fp32) → 2x less memory
- REDUCE_BLOCK_SIZE_N = 128 (vs 64 with fp32) → fewer reduce blocks
- Faster reduction with minimal precision impact
Inspired by leaderboard 'globals_patch' approach.

## Affected shapes
Only KSPLIT>1 shapes: M=16/K=7168 (KSPLIT=7 after get_splitk)
M=256/K=1536 has KSPLIT→1 after get_splitk, so NOT affected.

## Results
- First attempt: AttributeError (y.dtype when y=None) - bug in aiter's bf16 splitk
- Second attempt (pre-alloc y): Testing FAILED - precision issues from 7-way
  bf16 reduction. The accumulated rounding error exceeds rtol=1e-02.

The _USE_GEMM_SPLITK_BF16 flag is disabled by default for a reason:
the bf16 accumulation loses too much precision for FP4 GEMM with high KSPLIT.
