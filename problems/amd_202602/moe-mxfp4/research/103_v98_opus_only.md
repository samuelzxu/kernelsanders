# v103: v98 + Opus Sorting (Revert Bad E=257 bs=512 cktile from v100)

## Problem with v100
v100 extended cktile to E=257 bs=512 (tok/exp=15.9, inter_dim=256).
Result: E=257 bs=512 got WORSE: 284µs vs 244µs (+16% slower).

cktile (BF16) is slower than default CK (FP4) for E=257 bs=512 because:
- tok/exp=15.9 is moderate density → compute matters → FP4 MFMA 2x faster than BF16
- The quant savings (~10µs) don't compensate for slower GEMM

## Changes from v98
1. Revert cktile conditions to v98 (remove bad "tok/exp<20 and inter_dim<=256" condition)
2. Keep opus sorting (moe_sorting_opus_fwd if available) from v100

## Cktile Conditions (same as v98)
- tok/exp < 5 → cktile sk=2 (E=257 bs=16/128)
- tok/exp < 40 AND expert<=33 → cktile sk=1 (E=33 bs=16/128)
- Everything else → default CK FP4 path

## Expected Impact
- Should restore E=257 bs=512 to ~244µs
- Opus sorting may give marginal improvement on sorting-dominated shapes
