# Attempt 176: Pre-computed kvi with .clone() to fix correctness

## Problem
Attempt 173 gave 69.5µs but had ~25% correctness failure rate.
The _KVI_SHORT[:n] view shares memory with the module-level tensor.
torch.compile's internal memory management may overwrite this memory.

## Fix
Use .clone() instead of direct slice:
- `kvi = _KVI_SHORT[:n].clone()` creates an independent copy
- Adds ~0.5µs for memcpy (small: 256KB for bs=64, 1MB for bs=256)
- But eliminates memory aliasing risk
- Still saves the torch.arange GPU kernel launch (~2µs)
- Net: saves ~1.5µs per kv<=1024 assembly call
