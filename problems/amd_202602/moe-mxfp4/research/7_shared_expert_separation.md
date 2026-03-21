# Attempt 7: Shared Expert Separation

## Hypothesis
The shared expert is always selected for ALL tokens with weight=1.0.
Instead of routing it through MoE machinery, we could:
1. Extract shared expert weights from the weight tensors
2. Compute shared expert as a dense GEMM (no routing overhead)
3. Compute routed experts separately (now 8 per token instead of 9)
4. Add the results

## Implementation Plan
```python
# Separate shared expert (index = n_routed_experts)
shared_expert_idx = config["n_routed_experts"]  # e.g., 256 for DeepSeek-R1

# Extract routed expert selections (first n_experts_per_token columns)
routed_topk_ids = topk_ids[:, :config["n_experts_per_token"]]
routed_topk_weights = topk_weights[:, :config["n_experts_per_token"]]

# Compute routed experts only
routed_output = fused_moe(
    hidden_states,
    gate_up_weight_shuffled,
    down_weight_shuffled,
    routed_topk_weights,
    routed_topk_ids,
    ...
)

# Compute shared expert as dense GEMM
# Need to extract shared expert weights and compute:
# gate_up = x @ W_gate_up.T -> apply SwiGLU -> down = intermediate @ W_down.T
# Then add with weight=1.0

output = routed_output + shared_expert_output
```

## Potential Issues
1. Extra kernel launches may add overhead
2. Need to extract and process shared expert weights separately
3. AITER might already optimize shared expert internally
4. Weight extraction from shuffled format is complex

## Status
Implementing...
