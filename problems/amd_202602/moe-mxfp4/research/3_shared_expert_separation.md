# Attempt 3: Shared Expert Separation

## Hypothesis
The shared expert is always selected for ALL tokens with weight=1.0. Instead of routing it through MoE machinery, we could:
1. Compute shared expert as a dense GEMM (no routing overhead)
2. Compute routed experts separately
3. Add results together

This might reduce overhead from the MoE kernel having to handle the special "always selected" case.

## Analysis needed
- How much time is spent on shared expert vs routed experts?
- Can we use a different/faster kernel for the shared expert?
- Is the overhead of two kernels less than the MoE routing overhead?

## Potential issue
AITER's fused_moe may already handle this efficiently internally.
The shared expert is indexed as `n_routed_experts` (e.g., index 256 for DeepSeek-R1).

## Status
Need to check if this is feasible with the available AITER APIs.
