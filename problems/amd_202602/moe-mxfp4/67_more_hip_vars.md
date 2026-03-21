# v67: More HIP/HSA Env Vars

## Changes from v65
- `GPU_MAX_HW_QUEUES=1` (was 2) — fewer queues = less queue mgmt overhead
- `HSA_XNACK=0` — disable page fault handling for less dispatch overhead

## Kept from v65
- `HIP_FORCE_DEV_KERNARG=1` — device kernel args (proven 2-4% improvement)
- `@torch.inference_mode()` — autograd bypass
