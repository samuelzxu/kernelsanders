# #91 Optimal Tile Configs Based on CU Occupancy Analysis

## Key Insights
1. Shape 6 (M=256, K=1536): KSPLIT=2 triggers 768-block reduction kernel (3 waves!).
   BSM=32 KSPLIT=1 → same 384 tiles but no reduction. Expected savings: 3-4µs.
2. Shapes 3/4 (M=32, K=512): Extend fused kernel path from M<=16 to M<=32.
   Saves dynamic_mxfp4_quant(A) call (~1µs).
3. Shape 5 (M=64, K=2048): BSM=16 BSN=32 → 896 tiles (3.5 waves) vs 448 (1.75 waves).
   More tiles = better CU occupancy, fewer idle CUs in tail wave.
4. Shape 2 (M=16, K=7168): num_stages=3 waves_per_eu=2 for better pipelining.
5. All: torch.inference_mode() + output pre-allocation.

## Tile Count Analysis (256 CUs)
| Shape | Old Tiles | New Tiles | Old KSPLIT | New KSPLIT | Reduction |
|-------|-----------|-----------|------------|------------|-----------|
| M=4,N=2880,K=512 | 45 | 45 | 1 | 1 | None |
| M=16,N=2112,K=7168 | 264 | 264 | 8 | 8 | Same |
| M=32,N=4096,K=512 | 64 | 64 (fused) | 1 | 1 | None |
| M=32,N=2880,K=512 | 45 | 45 (fused) | 1 | 1 | None |
| M=64,N=7168,K=2048 | 448 | 896 | 1 | 1 | None |
| M=256,N=3072,K=1536 | 384+768red | 384 | 2 | 1 | ELIMINATED |

## Results
TBD
