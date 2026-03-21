# v73: Wider Selective ksplit

## Changes from v72
- Use tokens_per_expert < 64 (was: expert==33 && token<=128)
- This also covers E=257 bs=16 (0.6 tok/exp) which should benefit!
- Still restricted to inter_dim <= 1024 (d_expert <= 512)

## Shapes affected
- E=33 bs=16 d=512: 4.4 tok/exp → ksplit=2 (same as v72)
- E=33 bs=128 d=512: 35 tok/exp → ksplit=2 (same as v72)
- E=257 bs=16 d=256: 0.6 tok/exp → ksplit=2 (NEW!)
- E=257 bs=128 d=256: 4.5 tok/exp → ksplit=2 (NEW!)
