# v54: Tuned block_m for E=33 shapes

## Hypothesis
E=33 shapes have no tuned CSV config and use the default heuristic.
The default selects block_m=64 for bs=128 and block_m=128 for bs=512.
Force block_m=32 with non-temporal loads for all E=33 shapes.

## Rationale
- E=33 with topk=9: M*topk/E tokens per expert
  - bs=16: 4.4 tokens/expert → block_m=32 wastes less
  - bs=128: 35 tokens/expert → block_m=32 has 2 blocks, less padding
  - bs=512: 139 tokens/expert → block_m=32 has 5 blocks, fine granularity
- Non-temporal loads (use_nt=True) help when tokens per expert is small
  (data won't be reused, so cache bypass is beneficial)

## Previous findings
- v3: block_m=32 beat heuristic for E=33 bs=128 by ~5%
- v7: block_m=32 for all E=33 showed mixed results (within noise)
