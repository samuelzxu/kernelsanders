# v77: Higher split_k for Very Sparse Shapes

## Changes from v76
- split_k=4 for tokens_per_expert < 2 (E=257 bs=16: 0.6 tok/exp)
- split_k=2 for tokens_per_expert 2-5 (E=257 bs=128: 4.5 tok/exp)
- split_k=2 for E=33 moderate sparse
- No change for dense shapes

## Hypothesis
E=257 bs=16 has only 0.6 tokens per expert on average.
split_k=4 quadruples K-dimension parallelism, potentially filling more CUs.
But split_k=4 also needs 4x reduction vs 2x, which adds overhead.
