# v79: Extend cktile to ALL E=33 Shapes

## Changes from v78
- E=33 bs=512 d=512 (139 tok/exp): now uses cktile sk=1 (was standard CK)
- E=33 bs=512 d=2048: still standard CK (inter_dim=2048 > 1024 excluded)

## Hypothesis
cktile with split_k=1 eliminates quant without adding split overhead.
For bs=512 E=33 d=512, the quant kernels take ~20µs. If cktile bf16
is as fast as CK fp4 for the GEMM itself, we save the quant time.

## Risk
E=33 bs=512 has 139 tok/exp - very dense. The cktile kernel may be
slower than CK for dense scenarios because CK uses FP4 MFMA (2x compute
throughput) while cktile uses BF16 (lower throughput but no quant overhead).
