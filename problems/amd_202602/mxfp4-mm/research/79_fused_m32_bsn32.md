# #79 Fused quant+GEMM for M<=32 with BSN=32

## Hypothesis
The fused kernel failed for M=32 with BSN=64/BSK=512 due to register pressure.
With BSN=32, each threadblock handles half the N-tile → less registers needed.
BSK=512 with BSN=32: 32*512 fp4 elements per B tile = 8KB. With BSN=64 it was 16KB.
The fused kernel quantizes A in-register (needs bf16 A + fp4 A + scale) which
uses more registers than the separate quant path. Smaller BSN reduces the total.

## Key: "k512m32_bs32" from leaderboard — maybe fused kernel with BSN=32?
