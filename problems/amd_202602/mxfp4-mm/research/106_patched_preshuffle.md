# #106 Patched Preshuffle Kernel

## Approach
Monkey-patch the preshuffle kernel's JIT bug (NameError: 'b is not defined')
by modifying the source file on the runner before import. Add else branch
with masked loads.

## Results
Tests pass, but no performance improvement over #102:
- M=64: 20.5µs (vs #102's 20.0µs, slightly worse)
- M=256: 20.2µs (vs #102's 19.5µs, slightly worse)

The preshuffle variant adds overhead from:
1. shuffle_scales(B_scale): ~0.5µs per call
2. shuffle_scales(A_scale): ~0.5µs per call
3. Triton recompilation (cache cleared for patch)

The (N//16, K*16) weight tiling doesn't provide enough speedup
to offset the scale shuffling overhead.

## Conclusion
Preshuffle is not faster than basic gemm_afp4wfp4 for our shapes.
The basic variant already has remap_xcd and tuned configs.
