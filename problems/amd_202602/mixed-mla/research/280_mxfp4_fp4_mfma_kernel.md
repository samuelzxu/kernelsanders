# Attempt 280: MXFP4 FP4 MFMA Flash-Decode Kernel

## Summary
First working FP4 MFMA attention kernel on AMD MI355X (gfx950).
Uses `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` for QK^T and
bf16 MFMA for PV with in-kernel V dequantization from FP4.

## Key Results
- Correctness: PASSES all 4 test configs within rtol/atol=1e-2
- Performance: 5-7x slower than assembly (memory scheduling bottleneck)
- Best version (v10): bs=256/kv=8192 at 2290us vs assembly 309us

## Optimization History
| Version | bs=256/kv=8192 | Key change |
|---------|---------------|------------|
| v1 (1-wave, scalar) | 11,900us | Baseline |
| v4 (4-wave, wave0 compute) | 3,260us | Multi-wave cooperative loading |
| v9 (flat int32 copy) | 3,790us | Avoid div/mod (worse due to poor thread util) |
| v10 (buffer_load_lds) | 2,290us | Direct HBM→LDS transfer |
| v13 (double-buffer) | worse | Sequential barriers add overhead |

## Key Technical Findings

### FP4 MFMA intrinsic works with dynamic_mxfp4_quant data
- `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 4, 4, 0, sa, 0, sb)`
- Atype=4 (FP4), Btype=4 (FP4), sa/sb are E8M0 scale bytes
- Data from `dynamic_mxfp4_quant` is directly compatible — no scale shuffling needed
- 576/128 = 4.5 → 5 MFMA calls for QK^T (last padded with zeros)

### buffer_load_lds compiler fix
- `i32x4_t = int __attribute__((ext_vector_type(4)))` SRD type
- `reinterpret_cast` from struct to i32x4_t causes "Cannot select: bitcast" LLVM error
- FIX: construct via compound literal: `(i32x4_t){lo, hi, range, config}`
- Intrinsic: `llvm_amdgcn_raw_buffer_load_lds(srd, nullptr, 16, voffset, soffset, 0, 0)`
- M0 register set via `asm volatile("s_mov_b32 m0, %0" :: "s"(lds_addr))`

### tl.dot_scaled (Triton) limitations
- Requires BOTH operands from `_mxfp4_quant_op` — `tl.load` data has different packing
- Error: "Reduction dimension should pack the same number of elements"
- `_mxfp4_quant_op` cannot be called inside a Triton for-loop (compiler error)
- Conclusion: Triton path for MXFP4 attention is blocked without fixing packing

### V dequantization
- V = first 512 dims of KV, dequanted from FP4 to bf16 in LDS
- Lookup table: fp4_lut[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
- E8M0 scale: `__uint_as_float((uint32_t)e8m0 << 23)` = 2^(e8m0-127)
- Cooperative: all 256 threads, ~0.15us per tile

### Memory scheduling gap
- Assembly achieves ~61% of 6.4 TB/s HBM via 8-wave ping-pong
- Our kernel achieves ~8-10% with 4-wave cooperative loading
- The gap is NOT in compute (FP4 MFMA is fast) — it's in HBM bandwidth utilization
- Needs `__builtin_amdgcn_sched_group_barrier` for true compute/memory overlap

## LDS Layout (v10)
- Q fp4: NH * DQ_PACKED = 4608 bytes (no padding, contiguous)
- Q scale: NH * 18 = 288 bytes
- KV fp4: 32 * 288 = 9216 bytes
- KV scale: 32 * 18 = 576 bytes
- V bf16: 32 * 520 * 2 = 33280 bytes (padded stride for bank conflicts)
- P: 16 * 40 * 2 = 1280 bytes
- Total: ~49 KB (< 64 KB)

## Architecture
- Grid: (bs, num_splits)
- Block: 256 threads (4 wavefronts)
- Wave 0: all MFMA compute (QK^T + softmax + PV)
- Waves 1-3: cooperate on loading, idle during compute
- All waves: cooperative V dequant
- Tile size: 32 KV tokens
- Split-K with separate reduce kernel
