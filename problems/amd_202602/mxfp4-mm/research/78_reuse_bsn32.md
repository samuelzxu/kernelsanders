# #78 Reuse B_q + BSN=32 for K=512

## Inspiration
Leaderboard submission name: `submission_reuse_triton_k512m32_bs32.py` (~13µs)

## Changes from #53
1. BSN=32 instead of BSN=64 for K=512/M=32 shapes (more thread blocks, better XCD utilization)
2. Unshuffle B_scale for ALL K values (reuse B_q, no double quant) — "reuse" interpretation
3. BSN=32 for M=4 too (consistency)

## Hypothesis
- BSN=32 gives 2x more thread blocks → better 8-XCD distribution
- Reusing B_q via unshuffle eliminates dynamic_mxfp4_quant(B) entirely
- The unshuffle overhead (~2µs) might be offset by BSN=32's better parallelism
