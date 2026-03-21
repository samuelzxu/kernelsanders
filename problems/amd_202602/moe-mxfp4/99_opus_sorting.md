# v99: Opus Sorting Implementation

## Approach
Same as v97 but uses `aiter.moe_sorting_opus_fwd` instead of
`aiter.moe_sorting_fwd` for the sorting step.

AITER has two sorting implementations:
- Default: standard sorting with configurable dispatch policy
- Opus: alternative implementation (activated via AITER_USE_OPUS_MOE_SORTING=1)

The Opus implementation may be faster for certain expert/token distributions.

## Changes from v97
- Uses `moe_sorting_opus_fwd` instead of `moe_sorting_fwd`
