# v74: Bypass CSV Configs for Universal ksplit

## Approach
- `AITER_BYPASS_TUNE_CONFIG=1` forces default heuristic path for ALL shapes
- This allows `get_ksplit` override to apply to E=257 shapes too
- ksplit=2 when tokens_per_expert < 40 and inter_dim <= 1024

## Shapes affected (NEW vs v72)
- E=257 bs=16 d=256: 0.6 tok/exp → ksplit=2 (NEW!)
- E=257 bs=128 d=256: 4.5 tok/exp → ksplit=2 (NEW!)
- E=33 bs=16 d=512: 4.4 tok/exp → ksplit=2 (same as v72)
- E=33 bs=128 d=512: 35 tok/exp → ksplit=2 (same as v72)

## Risk
- E=257 shapes lose tuned kernelName1/2 (use auto-select instead)
- E=257 large batches may be slightly slower without tuned configs
- E=33 shapes are unaffected (no CSV config exists)
