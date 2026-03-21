# #112 Unified Kernel: Fused Quant + GEMM + Atomic Add

## Approach
Single @triton.jit kernel for ALL shapes:
- Fuses bf16→fp4 quant into GEMM (eliminates separate quant kernel)
- Uses remap_xcd for XCD-aware scheduling
- Uses tl.atomic_add for split-K reduction (eliminates separate reduce kernel)
- Pre-zeros output for KSPLIT>1

## Kernel launch savings vs #102:
| Shape | #102 launches | #112 launches |
|-------|--------------|--------------|
| M=4 K=512 | 1 (fused) | 1 (unified) |
| M=16 K=7168 | 2 (quant+GEMM) + 1 (reduce) | 1 (unified) |
| M=32 K=512 | 2 (quant+GEMM) | 1 (unified) |
| M=64 K=2048 | 2 (quant+GEMM) | 1 (unified) |
| M=256 K=1536 | 2 (quant+GEMM) + 1 (reduce) | 1 (unified) |

## Risks
- atomic_add contention for KSPLIT>1 (multiple writes to same output location)
- y.zero_() adds overhead for KSPLIT>1 shapes
- Fused quant adds register pressure
- Single kernel = one JIT compilation per shape (no reuse)

## Results
FAILED correctness for KSPLIT>1 shapes (M=8 N=2112 K=7168).
atomic_add on bf16 with 7 K-splits gives non-deterministic accumulation order,
causing ~1.5% relative error (exceeds rtol=1e-02 tolerance).
Example: -2.03125 vs -2.0625 (0.03125 difference on ~2.0 value)
KSPLIT=1 shapes (M=64, M=256) passed correctly.

Fix options: fp32 atomic_add + bf16 conversion, or separate reduce kernel.
Both add complexity/overhead that negates the benefit.
