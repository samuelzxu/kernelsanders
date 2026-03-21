# Attempt 165: Rotating buffer TunableOp + nks=16 for bs=64/kv=1024

## Changes
1. PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=256 (MiB)
   - Rotates inputs during tuning for cold-cache realism
2. nks=16 for bs<=64/kv<=1024 (was 8)
   - bs=64: 1024 CTAs = 4 waves (was 512 = 2 waves)
