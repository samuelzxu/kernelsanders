# #94 No-Reduction K=1536

## Hypothesis
BSM=32 BSN=64 BSK=1024 KSPLIT=1 for K=1536/M=256.
Same 384 tiles as KSPLIT=2 but no 768-block reduction kernel.
BSK=1024 with K=1536: 2 iterations (1024+512 with EVEN_K=False masking).

## Results
FAILED - NaN values at row 84 for M=256 shape.
BSK=1024 with K=1536 causes out-of-bounds access:
- K_packed = 768, BSK_packed = 512
- Second iteration accesses [512, 1024) but only [512, 768) is valid
- EVEN_K=False masking has a bug with FP4 packed data

## Conclusion
Cannot use BSK > K for this kernel without fixing the masking logic.
BSK must evenly divide K for correctness.
K=1536 requires BSK in {128, 256, 512} (all divide 1536).
