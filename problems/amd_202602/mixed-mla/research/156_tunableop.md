# Attempt 156: PYTORCH_TUNABLEOP_ENABLED for GEMM auto-tuning

## Discovery
PYTORCH_TUNABLEOP_ENABLED=1 routes all GEMM calls through a tuning layer
that benchmarks all rocBLAS and hipBLASLt solutions. Has shown ~22% improvement
in benchmark tests.

## Changes vs 153
- Add PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=1 env vars
- Combined with torch.compile for kv<=1024 GEMM from 153

## Expected Impact
- Tuning happens during warmup, cached for ranked runs
- May improve bs=4/kv=8192 uncompiled GEMM (currently 42.5µs)
- May also improve the compiled GEMM kernels if torch.compile uses BLAS underneath
- Risk: tuning timeout if too many solutions to benchmark
