# #153 - XCD-Aligned GROUP_SIZE_M + KSPLIT=2 for K=2048

## Hypothesis

Two independent improvements for the K=2048 shapes (N=7168):

### 1. GROUP_SIZE_M=8 for large-M (XCD alignment)

MI355X has 8 XCDs. The `remap_xcd` tile dispatch assigns tiles round-robin
across XCDs. For L2 cache reuse, tiles in the same "group" should share their
N-tile so they all read the same columns of B from the same XCD's L2.

N=7168, BLOCK_N=128 → num_pid_n = 56.

| GROUP_SIZE_M | tiles per group | tiles per XCD | B L2 reuse |
|---|---|---|---|
| 2 (current) | 2×56=112 | 14 | low |
| 8 (this exp) | 8×56=448 | 56 | high |

With GROUP_SIZE_M=8, each group spans 8 M-tiles at the same N-column, so
all 8 wavefronts read the same B tiles and benefit from L2 cache hits within
one XCD. This precisely matches the 8-XCD count for clean round-robin.

### 2. KSPLIT=2 for M_LEQ_64

For M=32-64 with K=2048, N=7168 with NUM_KSPLIT=1:
- M=32: 112 CTAs (44% GPU utilisation for 256 CUs)
- M=64: 224 CTAs (88% GPU utilisation)

With KSPLIT=2:
- M=32: 224 CTAs (88% → better latency hiding)
- M=64: 448 CTAs (175% → multiple waves per CU)

The KSPLIT reduction kernel is trivially cheap for these small M shapes.

## Changes from #148

- K=2048 "any": GROUP_SIZE_M 2 → 8
- K=2048 M_LEQ_64: NUM_KSPLIT 1 → 2

All other configs and O1 patch unchanged.
