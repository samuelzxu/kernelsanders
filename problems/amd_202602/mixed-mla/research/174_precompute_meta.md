# Attempt 174: Pre-computed metadata info at module level

## Changes vs 173
- Pre-compute get_mla_metadata_info_v1 results for all 5 assembly configs
- Saves JIT function call overhead (~1-2µs per assembly call)
- Combined with pre-computed kvi from 173
