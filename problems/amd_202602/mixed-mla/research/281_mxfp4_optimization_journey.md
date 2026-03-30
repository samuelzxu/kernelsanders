# Attempt 281: MXFP4 FP4 MFMA Kernel — Full Optimization Journey

## Best result: v10 at 2290us for bs=256/kv=8192 (assembly: 309us, 7.4x gap)

## Key Discoveries

### MI355X has 160KB LDS (not 64KB!)
- `MAX_SHARED_MEMORY = 160000` in HipKittens
- This enables true double-buffered V_bf16 (2 × 33KB = 66KB)
- CONTEXT.md incorrectly states 64KB

### MI355X has 8.0 TB/s HBM (not 6.4 TB/s!)
- HipKittens blog confirms 8.0 TB/s
- Assembly at 309us achieves ~49% BW utilization (not 61%)
- MXFP4 theoretical minimum: 80us (not 100us)

### FP4 MFMA works with dynamic_mxfp4_quant data
- `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` with Atype=4, Btype=4
- SRD construction: `(i32x4_t){lo, hi, range, 0x00110000}` (compound literal, no bitcast)
- No scale shuffling needed for raw HIP (unlike Triton tl.dot_scaled)
- 576/128 = 4.5 → 5 FP4 MFMAs for QK^T (last padded)

### Triton tl.dot_scaled blocked by two issues
1. Packing mismatch: `_mxfp4_quant_op` packing ≠ `tl.load` packing
2. `_mxfp4_quant_op` can't run inside a Triton for-loop

### buffer_load_lds works
- SRD via compound literal avoids LLVM bitcast error
- Each call: 64 lanes × 16 bytes = 1024 bytes per wavefront
- M0 register via `asm volatile("s_mov_b32 m0, %0" :: "s"(val))`
- `to_sgpr()` via `__builtin_amdgcn_readfirstlane`

## Version History (bs=256/kv=8192)

| Version | Time | Change | Why faster/slower |
|---------|------|--------|-------------------|
| v1 (1-wave scalar) | 11,900us | Baseline | - |
| v3 (4-wave redundant MFMA) | 6,880us | 4x load BW | Redundant MFMA wastes matrix unit |
| v4 (4-wave wave0 compute) | 3,260us | No redundant MFMA | 3 waves idle during compute |
| v5 (row-loop int32) | timeout | Avoid div/mod | Poor thread utilization |
| v9 (flat int32 copy) | 3,790us | Flat contiguous copy | Slightly worse than v4 |
| **v10 (buffer_load_lds)** | **2,290us** | Direct HBM→LDS | **BEST** |
| v13 (double-buffer 64KB) | worse | Sequential barriers | Extra sync overhead |
| v15 (async prefetch) | 2,820us | Prefetch next tile | s_waitcnt still serializes |
| v16 (4-wave all compute) | 5,540us | Redundant MFMA | 4x matrix unit contention |
| v17 (160KB true DB) | 4,140us | Double-buffer V | s_waitcnt + __syncthreads serialize |
| 8-wave sched_barrier | 3,820us | HipKittens-style | Wrong kernel architecture |
| 4-wave global V | 5,060us | V from global | Scattered byte reads kill perf |

## Structural Bottleneck

The V dequant (FP4→bf16, 33KB written to LDS per tile) requires `__syncthreads` for
LDS coherence before PV can read it. This serializes the pipeline. Assembly avoids this
because fp8 KV feeds directly to MFMA without dequantization.

## What would close the gap

1. **Eliminate __syncthreads from hot loop** — requires HipKittens cluster architecture
2. **Pin Register** — bypass hipcc's AGPR→VGPR→AGPR moves, use AGPRs as MFMA inputs
3. **XCD-aware grid scheduling** — balance L2/LLC cache hierarchy
4. **8-wave ping-pong with true role-swapping** — not redundant compute

These require essentially reimplementing the HipKittens framework for MLA attention.
