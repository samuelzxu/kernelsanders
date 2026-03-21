# Attempt 138: Adaptive nks targeting ~256 total work items (1 wave)

## Key Insight: Reduce Buffer Overhead
With nks=32 for bs=256/kv=8192:
- lg = bs*nks*NH*DV*4 = 256*32*16*512*4 = 268MB float32
- reduce reads 268MB + writes 4MB output
- 268MB at 5.3TB/s = 50µs just for reduce!

Reducing nks cuts the reduce overhead quadratically.

## MI355X Optimal: 1 Wave = 256 Work Items
With 256 CUs, aiming for 256 total CTAs = 1 full wave:
- No tail effect (no partial waves)
- Minimal scheduling overhead
- Smaller lg/ls buffers → faster reduce

## Changes vs 137
kv>1024 assembly path:
- bs=32: nks=8 (32*8=256=1 wave)   was 32
- bs=64: nks=4 (64*4=256=1 wave)   was 32
- bs=256: nks=4 (256*4=1024=4 waves) was 32 [conservative lower bound]

kv<=1024 assembly path:
- nks=4 (was 8)
- bs=64: 64*4=256=1 wave ✓
- bs=256: 256*4=1024=4 waves

## Data movement impact (bs=256/kv=8192)
Old (nks=32): KV reads=1.2GB + lg write=268MB + lg read=268MB = 1.74GB → ~328µs ideal
New (nks=4):  KV reads=1.2GB + lg write=34MB  + lg read=34MB  = 1.27GB → ~239µs ideal

Each CTA with nks=4/kv=8192: handles 2048 tokens (32 groups at kvg=64) — should be within kernel limits.

## Expected Impact
- bs=32/kv=8192: 96µs → ~70µs (nks 32→8, 1 wave, 4x smaller reduce)
- bs=64/kv=8192: 154µs → ~110µs (nks 32→4, 1 wave, 8x smaller reduce)
- bs=256/kv=8192: 350µs → ~260µs (nks 32→4, 4 waves, 8x smaller reduce)
- bs=64/kv=1024: 50µs → ~45µs (nks 8→4, 1 wave)
- bs=256/kv=1024: 115µs → ~100µs (nks 8→4, 4 waves, 2x smaller reduce)
