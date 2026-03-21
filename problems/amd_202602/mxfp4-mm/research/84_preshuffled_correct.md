# #84 preshuffled_scales with ATOM-correct format

## Key difference from #18
#18 failed because we passed B_scale_sh in (N, K//32) format.
Now we know from ATOM's code the correct format:
  B_scale_fmt = B_scale_sh.view(uint8).view(N//32, -1)
This gives (padded_N//32, K) format which the kernel expects.

## What this eliminates for M>=32
- dynamic_mxfp4_quant(B): NOT needed (reuse B_q + B_scale_sh)
- e8m0_unshuffle: NOT needed (kernel does in-kernel unshuffle)
- Only need: quant_A + e8m0_shuffle(A_scale) + reshape + GEMM

## The preshuffled_scales kernel
- Does in-kernel scale unshuffle via reshape+permute in the main loop
- Uses (X//32, K) scale format where groups of 32 rows share K scales
- Requires M >= 32 (assertion in wrapper)

## Risk
- Previous attempt (#18) failed with wrong results starting at column 32
- The scale format might still be wrong despite ATOM's pattern
