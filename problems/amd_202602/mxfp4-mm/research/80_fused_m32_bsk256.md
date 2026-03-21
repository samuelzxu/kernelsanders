# #80 Fused quant+GEMM M<=32 with BSK=256, BSN=32

## Hypothesis
Previous fused M=32 attempts failed with BSK=512 (register pressure).
BSK=256 halves the register usage per K-block iteration:
- bf16 A tile: 32×256 = 8192 elements (was 32×512 = 16384)
- fp4 B tile: 128×32 = 4096 elements (was 128×64 = 8192)
- Scale tiles proportionally smaller

With BSN=32 (from "bs32" leaderboard hint), the N tile is also smaller.
Total per-block register footprint: ~50% of the BSK=512/BSN=64 version.

## Key change
For K=512/M=32: BSK=256 (2 iterations), BSN=32 (90 N-blocks for N=2880)
Fused kernel handles M<=32 instead of just M<=16.
