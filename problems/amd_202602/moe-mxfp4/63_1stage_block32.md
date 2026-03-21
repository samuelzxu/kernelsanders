# v63: Force 1-Stage with block_m=32

## Root Cause Analysis of v59/v60 Crash
The 1-stage kernel name `2tg_ps_32x256` has `32x` meaning block_m=32.
But AITER's heuristic selects block_m=64 for E=33 bs=128 (34 tokens/expert).
This mismatch means `moe_sorting` creates sorted arrays with 64-token blocks,
but the kernel reads them assuming 32-token blocks → memory access fault.

## Fix
Force `md.block_m = 32` for ALL shapes when using 1-stage.
This ensures sorting granularity matches the kernel's tile size.

## Expected Result
- No more memory faults on E=33 shapes
- 1-stage for all 7 benchmark shapes
- ~27% speedup vs 2-stage (eliminates sorting + quant overhead)
