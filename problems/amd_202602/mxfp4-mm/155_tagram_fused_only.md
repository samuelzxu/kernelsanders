# #155 - TagRAM Padding, Fused-Path Only (Correctness Fix for #152)

## Problem with #152

#152 applied TagRAM padding globally before dispatching to either the fused
kernel or gemm_afp4wfp4. The padded B_q (non-contiguous stride) was passed to
gemm_afp4wfp4, which assumes contiguous B_q layout → massive corruption for
k=7168, m=16, n=2112 (33686/33792 elements wrong).

## Fix

Apply TagRAM padding **only in the fused kernel path** where we pass explicit
stride parameters to the Triton kernel. The fused kernel computes:

  b_ptrs = b_ptr + offs_k * stride_bk + offs_n * stride_bn

where stride_bk and stride_bn come from B_q_t.stride(). With the padded view,
stride_bn changes from K//2 to K//2+64. This correctly distributes addresses
across 8 L2 cache sets instead of 1.

The gemm_afp4wfp4 path continues to use the original contiguous B_q_uint8.

## Affected Shapes

The fused path is used when `m <= 16 AND NUM_KSPLIT == 1`:
- K=512 (N=2880, N=4096): K//2=256, 256%512=256≠0, no padding needed anyway
- K=2048 (N=7168), M=1-16: K//2=1024=2×512 → padding applied
  - stride changes from 1024→1088, 8× more L2 cache sets for B_q accesses

## Expected Impact

Moderate for M=1-16 K=2048 shapes. The fused kernel is already fast and may
be MFMA-compute-bound rather than memory-bandwidth-bound for these sizes.
Best-case: 5-10% improvement for those specific shapes.

## Changes from #148

- Fused kernel path: B_q_t uses padded stride(0)=K//2+64 for K=2048
- All other paths and configs: identical to #148
