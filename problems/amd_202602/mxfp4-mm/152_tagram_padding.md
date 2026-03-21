# #152 - TagRAM Stride Padding

## Hypothesis

AMD MI300X/MI355X L2 cache has a known TagRAM channel hotspotting issue when
matrix row strides are multiples of 512 bytes (documented in AMD CDNA tuning guides).

The benchmark shapes map to B_q (FP4 weights, uint8, shape N×K//2):

| Shape      | B_q row stride | Stride % 512 | Cache sets used |
|-----------|---------------|--------------|-----------------|
| K=512     | 256 bytes      | 256          | 2               |
| K=1536    | 768 bytes      | 256          | 2               |
| K=2048    | 1024 bytes     | **0**        | **1 (HOTSPOT)** |
| K=7168    | 3584 bytes     | **0**        | **1 (HOTSPOT)** |

When all rows of B_q hash to the same L2 TagRAM bank, wavefronts serialize
on L2 lookups, degrading memory bandwidth for the K=2048 and K=7168 shapes.

## Fix

Allocate a padded B_q buffer with row stride = K//2 + 64 bytes, breaking the
512-byte alignment. After padding, the 8 distinct offsets used by consecutive
rows (mod 512) are: 0, 64, 128, 192, 256, 320, 384, 448 — fully distributed.

The padded B_q is cached per unique B data pointer so the single copy is
paid only on the first (warmup) iteration. All benchmark warm iterations
use the cached padded view.

## Expected Impact

- K=2048 (N=7168): 1 TagRAM bank → 8 banks. Expect 10-20% improvement for
  small-M shapes where the kernel is memory-bandwidth bound.
- K=7168 (N=2112): Same benefit for the large-K shape.
- Other shapes: unchanged (K//2 not multiple of 512, no padding applied).

## Base

#148 (LLVM O1 + all configs). All configs and O1 patch unchanged.
