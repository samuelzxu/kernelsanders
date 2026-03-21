# #90 Tuned Large-K Configs + inference_mode + output pre-alloc

## Hypothesis
Three large-K shapes dominate our geomean (~21µs each):
- M=16, N=2112, K=7168: KSPLIT=8→4 (less reduction overhead)
- M=64, N=7168, K=2048: BSN=32→64 (more N parallelism)
- M=256, N=3072, K=1536: KSPLIT=2→1 (no split-K overhead)

Also add:
- torch.inference_mode() decorator
- Pre-allocated output buffer caching

## Changes
- K=7168/M=16: BSN=128, KSPLIT=4 (was BSN=64, KSPLIT=8)
- K=2048/M=64: BSN=64, BSK=512 (was BSN=32, BSK=1024)
- K=1536/M=64,M=256: KSPLIT=1, BSM=128 (was KSPLIT=2, BSM=64)
- @torch.inference_mode() on custom_kernel
- Pre-allocate output buffer for fused kernel path

## Results (Ranked)
Mixed results:
- M=4, K=512: 12.4µs (unchanged)
- M=16, K=7168: 22.3µs (**worse** +1µs - KSPLIT=4 gives only 132 tiles vs 264)
- M=32, N=4096, K=512: 12.6µs (unchanged)
- M=32, N=2880, K=512: 12.5µs (unchanged)
- M=64, K=2048: 20.4µs (**better** -1.3µs - BSN=64 helps!)
- M=256, K=1536: 21.7µs (**worse** +1.1µs - BSM=128/KSPLIT=1 gives only 96 tiles)

KEY LEARNINGS:
- K=2048: BSN=64 is better than BSN=32 (more N parallelism matters)
- K=7168: KSPLIT=8 is better than KSPLIT=4 (264>132 tiles for CU occupancy)
- K=1536: KSPLIT=2 is needed (192 tiles without is too few for 256 CUs)
