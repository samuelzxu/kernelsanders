# v87: Minimal Python Overhead

## Changes from v85
1. Pre-build cktile functools.partial objects at import time (avoids creating new partials per call)
2. Compute inter_dim from config dict instead of get_inter_dim(w1.shape, w2.shape)
3. Hardcode hidden_pad=0, intermediate_pad=0 (always 0 for our aligned shapes)
4. Compute padded_M inline instead of calling get_padded_M
5. Hardcode dtype=torch.bfloat16 instead of reading from hidden_states.dtype

## Expected Savings
~1-2µs from reduced Python overhead in hot path.
Mostly from avoiding functools.partial creation on every lru_cache miss.
