# v75: Tuned ksplit Threshold

## Key Insight from v74
E=257 bs=16 d=256: tokens_per_expert=0.6 → 32% faster with ksplit=2!
E=257 bs=128 d=256: tokens_per_expert=4.5 → 16% faster!
E=257 bs=512 d=256: tokens_per_expert=17 → 7% SLOWER!

## New Threshold
- Very sparse (< 5 tok/exp): always ksplit=2
- Moderate sparse (5-40 tok/exp): only if inter_dim <= 512
- Dense (>= 40 tok/exp): never ksplit=2
- Large inter_dim (> 1024): never ksplit=2
