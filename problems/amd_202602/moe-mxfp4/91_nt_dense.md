# v91: Non-Temporal Loads for All Shapes

## Rationale (from AMDGPU Kernel Optimization Guide)
"For data that is streamed and does not need to be cached, consider
using non-temporal loads." MoE weights are large and accessed once
per expert per call. NT loads bypass L2 cache, freeing it for
activation data that IS reused between stage1 and stage2.

## Change
AITER_USE_NT=1 forces use_nt=True for ALL shapes (was: only tok/exp < 64).
This affects the CK stage1/stage2 kernels' use_non_temporal_load parameter.

## Expected Impact
Dense E=33 bs=512 shapes have large weights that may pollute L2.
NT loads should help by keeping L2 available for activations.
Sparse shapes already use NT loads (default heuristic), so no change there.
