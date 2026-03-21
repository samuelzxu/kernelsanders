# #154 - Fused Quant+GEMM Extended to M<=32

## Hypothesis

The fused quant+GEMM kernel (inline quantization of A inside the GEMM loop)
avoids one `dynamic_mxfp4_quant(A)` kernel dispatch per call. Each kernel
dispatch costs ~1-2µs in fixed overhead. For small M shapes where the GEMM
itself is fast, this overhead is a significant fraction of total runtime.

Currently the fused path is used only for M<=16. This experiment extends it
to M<=32, capturing savings for:
- M=32, K=512, N=2880 (config: M_LEQ_32, BLOCK_K=512, 1 loop iteration)
- M=32, K=512, N=4096 (config: M_LEQ_32, BLOCK_K=512, 1 loop iteration)

These are the best-case shapes for the extension:
- BLOCK_K=512 = full K, so only 1 K-loop iteration (no extra quant work)
- NUM_KSPLIT=1 (no split-K; guard keeps safety)

The M<=32 fused kernel with BLOCK_M=32, BLOCK_N=64, BLOCK_K=512 is already
verified to produce correct results (same tile size as the M_LEQ_32 config).

## Why Not Larger M?

For M=32 with K=2048 (BLOCK_K=512, 4 K-iterations): the extra quantization
work per iteration increases register pressure and may slow the MFMA pipeline.
The NUM_KSPLIT=1 guard naturally excludes K=1536/K=7168 (which have KSPLIT>=2).
K=2048 has KSPLIT=1 but the 4-iteration loop is borderline — excluded here.

## Changes from #148

- Fused kernel threshold: `m <= 16` → `m <= 32`
- All configs and O1 patch unchanged.
