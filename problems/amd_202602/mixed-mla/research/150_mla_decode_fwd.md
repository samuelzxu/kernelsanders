# Attempt 150: Use mla_decode_fwd high-level API

## Hypothesis
The mla_decode_fwd function from aiter.mla may:
1. Allocate lg/ls internally more efficiently (saving 2 allocation calls)
2. Have internal optimizations in how it dispatches stage1 + reduce
3. Potentially skip the reduce step when nks=1 or optimize for specific nks values

## Risk
- The function signature might differ from what we expect
- The internal allocation might be SLOWER (extra indirection)
- logit_cap=0.0 might add unnecessary branching
