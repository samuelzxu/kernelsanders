# #92 Combined Best Configs

## Changes vs #53 (previous best at 16.2µs geomean)
1. K=2048/M=64: BSN=64 BSK=512 (confirmed -1.3µs in #90/#91)
2. K=7168/M=16: num_stages=3 waves_per_eu=2 (small -0.3µs from #91)
3. torch.inference_mode() + output pre-alloc for fused path
4. Keep fused at M<=16 only (M<=32 fused was WORSE in #91)
5. K=1536: Keep original BSM=64 KSPLIT=2 (BSM=32 KSPLIT=2 was worse)

## Expected improvement
K=2048: -1.3µs, K=7168: -0.3µs = ~1.6µs total improvement
Expected geomean: ~15.5µs (from 16.2µs)

## Results (Ranked)
- M=4, K=512: 12.4µs (same)
- M=16, K=7168: 21.1µs (-0.2µs)
- M=32, N=4096: 12.7µs (same)
- M=32, N=2880: 12.6µs (same)
- M=64, K=2048: 20.5µs (-1.2µs ✓)
- M=256, K=1536: 20.6µs (same)
- Geomean: ~16.1µs (from 16.2µs)

CURRENT BEST SUBMISSION. K=2048 improvement confirmed.
