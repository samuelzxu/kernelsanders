# v98: Non-Temporal Loads for All Shapes

## Approach
Same as v97 (pre-alloc + cktile) but adds `AITER_USE_NT=1` to force
non-temporal loads for ALL shapes, including dense ones.

Default behavior: use_nt=True only when `tok_per_expert < 64`.
For dense E=33 bs=512 shapes (tok_per_expert=139), NT loads are OFF.

### Rationale
MoE weights are accessed once per call - they don't benefit from L2 caching.
NT loads bypass L2, freeing cache for activation data that IS reused between
stage1 and stage2. This could help dense shapes where weight data is large
(33 experts × large weight matrices ≈ 48MB FP4 data).

## Changes from v97
- Added `os.environ['AITER_USE_NT'] = '1'`
