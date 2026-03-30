# v51: 1-Stage Assembly Kernel Retry

## Hypothesis
On warm runners where CK 2-stage modules are already cached,
the module_moe_asm JIT build (~30s) should fit within the 12-min timeout.
Previous attempts timed out because BOTH CK 2-stage (~105s) AND
module_moe_asm (~30s) needed building simultaneously.

## Approach
- Force `run_1stage=True` for `inter_dim <= 1024` (d_expert <= 512)
- Keep 2-stage for d_expert=2048 (tiny precision issue: 2/229376 elements)
- The 1-stage `fmoe_g1u1` assembly kernel does:
  sort + quant + GEMM1 + SwiGLU + reQuant + GEMM2 + weighted_reduction
  ALL in one kernel launch

## Expected Improvement
- Eliminates sorting kernel (14% of GPU time)
- Eliminates quant kernels (13% of GPU time)
- Total: ~27% improvement on 6/7 benchmark shapes
- Only d_expert=2048 shape uses 2-stage (1/7 shapes)

## Risk
- Runner might not have cached modules → timeout at 12 min
- d_expert=2048 still uses 2-stage (no improvement there)
