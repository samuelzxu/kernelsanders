# v105: Combined v101 + v104

## Changes from v101
Adds block_m=64 override for E=33 bs=512 d=2048 (from v104).
Both optimizations target E=33 bs=512 but for different inter_dim ranges.

## Shape Coverage
| Shape | Condition | Action |
|-------|-----------|--------|
| E=257 bs=16 | tok/exp=0.56, inter_dim=256 | sk=2 cktile |
| E=257 bs=128 | tok/exp=4.48, inter_dim=256 | sk=2 cktile |
| E=257 bs=512 | tok/exp=17.9, inter_dim=256 | default CK (not expert<=33) |
| E=33 bs=16 | tok/exp=3.9, inter_dim=512 | sk=2 cktile |
| E=33 bs=128 | tok/exp=31, inter_dim=512 | sk=1 cktile |
| E=33 bs=512 d=512 | tok/exp=124, inter_dim=512 | **sk=1 cktile (v101)** |
| E=33 bs=512 d=2048 | tok/exp=124, inter_dim=2048 | **block_m=64 (v104)** |

## CU Utilization Analysis
### E=33 bs=512 d=512 (v101 part):
- Default CK block_m=64: 99/256 = 39% CU util
- cktile block_m=16: 297/256 = 116% CU util (full!)

### E=33 bs=512 d=2048 (v104 part):
- Default CK block_m=128: 66/256 = 26% CU util
- block_m=64: 99/256 = 39% CU util (+50%)

## Expected Improvement
- E=33 bs=512 d=512: potentially 10-30µs faster (v101 tested separately)
- E=33 bs=512 d=2048: potentially 5-10% faster from better CU util
- Combined geomean improvement vs v103 baseline (~153µs)
