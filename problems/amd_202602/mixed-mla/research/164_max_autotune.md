# Attempt 164: max-autotune + per-shape compile

## Change vs 163
torch.compile mode: default → "max-autotune"
Enables: CK/Triton/hipBLAS kernel search + HIP graph trees

## Risk
- max-autotune takes longer to compile (autotuning benchmarks)
- May blow test timeout (900s) if too many candidates
- HIP graph overhead if shapes change during benchmark
