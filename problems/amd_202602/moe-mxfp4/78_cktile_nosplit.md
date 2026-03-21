# v78: cktile with split_k=1 for E=33 Moderate Sparse

## Hypothesis
E=33 bs=128 (35 tok/exp): v72 used cktile split_k=2 → 106µs.
What if we use cktile split_k=1 (no splitting)?
- Still gets bf16 activation path (no quant overhead) via md.ksplit=2 flag
- But the actual cktile GEMM runs without K-dimension splitting
- Saves the split_k reduction kernel overhead

## Shapes
- E=257 bs=16: split_k=2 (very sparse, needs parallelism)
- E=257 bs=128: split_k=2 (sparse, needs parallelism)
- E=33 bs=16: split_k=2 (very sparse)
- E=33 bs=128: split_k=1 (moderate, try NO split) ← NEW
- All other: standard CK (unchanged)
