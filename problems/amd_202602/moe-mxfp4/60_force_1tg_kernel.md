# v60: Force 1-Stage with Specific Kernel Variant

## Problem (v59)
The 1-stage heuristic selected `2tg_ps_32x256` (sub_GU=256) for E=33 d=512.
This kernel crashes with memory fault because inter_dim=512 > sub_GU=256
requires 2 grid tiles, and something goes wrong with the 2tg variant.

## Fix
Force `fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512` for all shapes.
- `1tg` = 1 threadgroup (simpler, safer)
- `ps` = preshuffle
- `32x512` = block_m=32, sub_GU=512
- sub_GU=512 >= inter_dim for all benchmark shapes (max inter_dim=2048)

Wait - inter_dim=2048 > sub_GU=512! For d_expert=2048, inter_dim=2048.
With sub_GU=512: gdx = ceil(2048/512) = 4 tiles. This should work
as long as the 1tg kernel supports multi-tile (non-persistent mode).

## Triton Quant Override
Same as v59: replaces HIP quant with Triton quant to avoid module_quant build.
