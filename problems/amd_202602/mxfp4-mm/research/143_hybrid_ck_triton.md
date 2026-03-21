# #143 Hybrid CK ASM + Triton

## Approach
Use CK ASM (aiter.gemm_a4w4) for shapes with tuned CSV configs:
- M=64, N=7168, K=2048: CSV reports 6.8µs (vs Triton ~14µs isolated)
- M=256, N=3072, K=1536: CSV reports 6.2µs (vs Triton ~14µs isolated)

Use Triton (gemm_afp4wfp4) for shapes without CK configs:
- M=4/32 K=512: fused kernel or regular Triton
- M=16 K=7168: Triton with KSPLIT=7

## Difference from #88
#88 used CK ASM for ALL shapes → K=512 shapes had no tuned config → terrible.
This hybrid ONLY uses CK ASM where tuned configs exist.

## Overhead
CK ASM path needs: e8m0_shuffle(A_scale) + uses B_shuffle + B_scale_sh directly.
Additional ~0.5µs for e8m0_shuffle vs standard path.

## Results
M=64: 24.8µs (WORSE +5µs vs Triton's 19.8µs)
M=256: 23.3µs (WORSE +3.4µs vs Triton's 19.9µs)
Other shapes same (still use Triton).

The CSV's 6.8µs is KERNEL-ONLY time on warm data.
Ranked benchmark adds: cold cache + quant + scale shuffle overhead.
CK ASM is not competitive in practice for these shapes.
