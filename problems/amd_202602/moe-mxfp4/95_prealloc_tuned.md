# v95: Pre-allocated Sorting Buffers + Block_m Tuning

## Approach
Two independent optimizations combined:

### 1. Pre-allocated Sorting Buffers
The `moe_sorting()` function allocates 5 GPU tensors on every call:
- sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf

These sizes are deterministic per shape configuration (M, E, model_dim, block_size_M).
By pre-allocating and reusing these buffers, we eliminate 5 `torch.empty()` calls
per invocation (~1-3us per call on AMD GPUs).

We call `aiter.moe_sorting_fwd()` directly with our cached buffers instead of
going through the `moe_sorting()` wrapper.

### 2. Block_m Override for E=33 Dense Shapes
The default `get_block_size_M()` heuristic computes:
- E=33, bs=512, d=512: block_m=64
- E=33, bs=512, d=2048: block_m=128

The DSv3 tuned config uses block_m=32 for E=257 shapes (which are similar in
tokens_per_expert density). We try block_m=32 for E=33 d=512 dense shapes
to test if smaller tiles improve occupancy.

## Expected Impact
- Pre-allocation: ~2-5us savings per call (7 shapes = ~14-35us total)
- Block_m tuning: unclear, might help or hurt E=33 d=512 bs=512

## Changes from v85
- Added `_fast_sorting()` that caches and reuses GPU buffers
- Added block_m=32 override for E=33 d<=512 dense shapes
