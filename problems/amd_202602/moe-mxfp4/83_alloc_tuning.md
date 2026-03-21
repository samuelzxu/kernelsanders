# v83: PyTorch Memory Allocator Tuning

## Change from v78
- Added `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
- This reduces memory fragmentation overhead in the caching allocator
- May help with the many small allocations in fused_moe pipeline
