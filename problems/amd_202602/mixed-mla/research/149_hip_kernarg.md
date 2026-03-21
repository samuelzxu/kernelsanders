# Attempt 149: HIP_FORCE_DEV_KERNARG=1

## Discovery
Research found that `HIP_FORCE_DEV_KERNARG=1` environment variable puts HIP kernel
arguments directly to device memory, reducing 2-3µs latency per kernel launch.

## Potential Impact
Per assembly call: ~5 kernel launches (metadata, stage1, reduce, Q quant, arange)
Savings: 2-3µs × 5 = 10-15µs per call
For the 4 assembly configs, this could save 40-60µs total → ~5% geomean improvement

## Risk
- Environment variable might be ignored by the benchmark framework
- Might not work with the specific kernel dispatch path used by aiter
- Anti-cheat: this is a standard HIP runtime optimization, not a hack
