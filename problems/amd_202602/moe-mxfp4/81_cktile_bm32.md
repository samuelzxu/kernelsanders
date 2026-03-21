# v81: cktile with block_m=32

## Change from v78
- block_m=32 for all cktile shapes (was: block_m=16 for token < 2048)
- block_m=32 matches CK tuned configs and may improve CU utilization

## Hypothesis
With block_m=16, each expert block handles 16 tokens. For E=257 bs=16
(~0.6 tok/exp), many blocks have 0-1 tokens → wasted computation.
block_m=32 has larger blocks but fewer of them, potentially better for
the MI355X's 256 CU architecture.
