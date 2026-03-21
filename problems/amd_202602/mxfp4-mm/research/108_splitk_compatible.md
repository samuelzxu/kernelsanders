# #108 Split-K Compatible BSK Values

## Critical Discovery
get_splitk() silently adjusts configs! With our injected configs:
- K=1536 BSK=512 KSPLIT=2: get_splitk reduces to KSPLIT=1 (K_packed=768, 768%512≠0)
  → Only 192 tiles (0.75 utilization) instead of intended 384
- K=7168 BSK=256 KSPLIT=8: get_splitk reduces to KSPLIT=7 (SPLITK_BLOCK_SIZE//2=512, 3584%512=0 works for 7 but not 8)
  → 231 tiles instead of intended 264

## Fix
Choose BSK values where get_splitk PRESERVES the KSPLIT:
- K=1536: BSK=256 KSPLIT=2 → SPLITK_BLOCK_SIZE=768, 768%384=0 ✓ → 384 tiles
- K=7168: BSK=128 KSPLIT=8 → SPLITK_BLOCK_SIZE=896, 3584%448=0 ✓ → 264 tiles

## Expected impact
- K=1536: 2x more tiles (192→384), should reduce tail wave inefficiency
- K=7168: 14% more tiles (231→264)
- Trade-off: smaller BSK = more K iterations per tile (more loop overhead)

## Results (Ranked)
WORSE on both target shapes:
- K=7168 M=16: 22.7µs (+1.4µs) - BSK=128 doubles K iterations (56 vs 28)
- K=1536 M=256: 24.5µs (+4.5µs) - BSK=256/KSPLIT=2 adds iterations + reduction

## Key Learning
More tiles does NOT mean faster! The overhead from:
1. More K iterations per tile (smaller BSK = more inner loop iterations)
2. Additional reduction kernel launch for KSPLIT>1
3. Per-tile setup overhead
outweighs the CU occupancy improvement from more tiles.

The get_splitk() adjustment to KSPLIT=1 for K=1536 was actually OPTIMAL -
fewer iterations with larger tiles is better for this shape despite 0.75 utilization.

CONCLUSION: get_splitk() optimizations are correct. Don't fight them.
