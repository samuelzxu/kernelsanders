# v84: cktile sk=1 with block_m=64 for Dense E=33

## Key Difference from v79
v79 used block_m=16 for E=33 bs=512 → 232µs (10% worse than CK 211µs).
v84 uses block_m=64 for dense E=33 shapes (tok/exp >= 40).
block_m=64 matches the CK heuristic for this shape.

## Hypothesis
The 10% regression in v79 may have been from block_m=16 overhead
(too many small blocks for dense scenarios). With block_m=64, the
cktile kernel may be competitive while still eliminating quant overhead.
