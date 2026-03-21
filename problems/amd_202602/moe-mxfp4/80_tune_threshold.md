# v80: Tune split_k Threshold

## Changes from v78
- tok/exp < 1 → sk=2 (was: tok/exp < 5) — only E=257 bs=16 (0.6)
- tok/exp 1-5 → sk=1 (was: sk=2) — E=257 bs=128 (4.5), E=33 bs=16 (4.4)
- tok/exp 5-40 + E<=33 → sk=1 (unchanged) — E=33 bs=128 (35)

## Hypothesis
E=257 bs=128 (4.5 tok/exp) may benefit from sk=1 like E=33 bs=128 does.
The split_k reduction kernel adds overhead that may not be worth the parallelism
at 4.5 tokens per expert.
