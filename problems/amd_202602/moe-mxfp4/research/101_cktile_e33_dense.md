# v101: cktile sk=1 for E=33 bs=512 d=512 (Fixed - no bad E=257 condition)

## Problem
E=33 bs=512 d=512 takes ~211µs with default CK 2-stage.
Shape: tok/exp=124, inter_dim=512, block_m=64 (CK default)

## Key Insight: CU Utilization
Default CK for E=33 bs=512 d=512 uses block_m=64:
- M-blocks per expert: ceil(124/64) = 3 (with tok/exp=139 using topk=9)
- Wait: get_padded_M(512)=512, tokens_per_expert=512*9/33=139.6
- Total M-blocks: 33 × 3 = 99
- MI355X has 256 CUs: 99/256 = 39% CU utilization (61% idle!)

cktile with block_m=16:
- M-blocks per expert: ceil(139.6/16) = 9
- Total M-blocks: 33 × 9 = 297
- CU utilization: 297/256 = 1.16 waves (near-full utilization!)

This 3x improvement in CU utilization should significantly reduce idle time.

## Changes from v103
- Extended cktile condition: `inter_dim <= 512 AND tok/exp < 150 AND expert <= 33`
- NO E=257 bs=512 cktile (that was wrong in v100, proved +16% slower)
- Keeps: opus sorting, pre-alloc, NT loads, cktile for sparse E=257/E=33

## Shapes Affected
| Shape | tok/exp | inter_dim | expert | Action |
|-------|---------|-----------|--------|--------|
| E=257 bs=16 | 0.56 | 256 | 257 | sk=2 (tok/exp<5) |
| E=257 bs=128 | 4.48 | 256 | 257 | sk=2 (tok/exp<5) |
| E=257 bs=512 | 17.9 | 256 | 257 | **default CK** (not expert<=33) |
| E=33 bs=16 | 3.9 | 512 | 33 | sk=2 (tok/exp<5) |
| E=33 bs=128 | 31 | 512 | 33 | sk=1 (same as before) |
| E=33 bs=512 d=512 | 124 | 512 | 33 | **sk=1 NEW** |
| E=33 bs=512 d=2048 | 124 | 2048 | 33 | default CK (inter_dim>1024) |

## Expected Impact
- E=33 bs=512 d=512: ~10-30µs faster from quant savings + CU utilization
- Potential: 211µs → ~180-200µs
- Geomean improvement: ~2-5%
