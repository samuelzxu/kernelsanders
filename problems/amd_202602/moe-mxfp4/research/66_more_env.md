# v66: Additional ROCm Environment Variables

## New Variables (on top of v65)
- `HSA_ENABLE_SDMA=0`: Disables System DMA engine. Forces GPU copy engine for memory operations.
  May reduce overhead for small tensor allocations.
- `PYTORCH_TUNABLEOP_ENABLED=0`: Disables PyTorch tunable ops search, eliminating
  per-call overhead of checking for tunable alternatives.

## Kept from v65
- `HIP_FORCE_DEV_KERNARG=1`: Device kernel args (saves 2-3µs/launch)
- `GPU_MAX_HW_QUEUES=2`: Limits HW queues
- `@torch.inference_mode()`: Autograd bypass
