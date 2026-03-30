# v68: Optimal Environment Variables

## Variables
- `HIP_FORCE_DEV_KERNARG=1` — device kernel args (proven -2-4%)
- `GPU_MAX_HW_QUEUES=2` — optimal queue count (proven in v65)
- `AMD_DIRECT_DISPATCH=1` — direct kernel dispatch (default, ensure set)
- `HIP_INITIAL_DM_SIZE=67108864` — 64MB initial heap (reduce alloc overhead)

## Hypothesis
AMD_DIRECT_DISPATCH should already be 1 (default since ROCm 6.2).
HIP_INITIAL_DM_SIZE increase could help by pre-allocating device memory
heap, reducing dynamic allocation overhead during fused_moe execution.
