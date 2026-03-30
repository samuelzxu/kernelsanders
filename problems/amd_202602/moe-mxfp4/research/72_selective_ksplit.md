# v72: Selective ksplit for Small E=33 Shapes

## Approach
Override `get_ksplit` to return 2 for E=33 token<=128 inter_dim<=512.
This activates the cktile path (bf16 activations, no quant) for these shapes.

## v71 Results That Motivate This
- E=33 bs=16 d=512: 58µs vs 87µs (33% FASTER with cktile)
- E=33 bs=128 d=512: 106µs vs 122µs (13% FASTER)
- E=33 bs=512 d=512: 248µs vs 208µs (19% slower → keep standard)
- E=33 bs=512 d=2048: 680µs vs 328µs (107% slower → keep standard)

## Shapes Affected
- bs=16 E=33 d=512: ksplit=2 → cktile path (faster)
- bs=128 E=33 d=512: ksplit=2 → cktile path (faster)
- All other shapes: ksplit=0 → standard CK path (keep current speed)
