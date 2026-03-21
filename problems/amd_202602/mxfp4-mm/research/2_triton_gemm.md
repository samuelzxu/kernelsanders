# Attempt 2: Triton-based MXFP4 GEMM

## Hypothesis
The reference.py notes: "aiter also has other a4w4 implements using triton, https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py"

The Triton-based GEMM may have better autotuning for our shapes, or lower launch overhead for small-M cases.

## Changes
- Use `aiter.ops.triton.gemm.basic.gemm_afp4wfp4` instead of `aiter.gemm_a4w4`
- The Triton kernel may have different input format requirements

## Result
Pending...
