# Attempt 2: doweight_stage1=True

## What we tried
Changed `doweight_stage1` from `False` to `True`.

This parameter controls where routing weights are applied:
- `False`: Weights applied in stage 2 (down projection) reduction
- `True`: Weights applied in stage 1 (gate_up projection) output

## Hypothesis
Applying weights earlier (in stage 1) might reduce intermediate memory traffic or allow better fusion.

## Code change
```python
doweight_stage1=True,  # was False
```

## Expected kernel name change
From logs, baseline uses:
- Stage1: `MulRoutedWeight0` (weights NOT applied)
- Stage2: `MulRoutedWeight1` (weights applied)

With `doweight_stage1=True`, expect:
- Stage1: `MulRoutedWeight1` (weights applied)
- Stage2: `MulRoutedWeight0` (weights NOT applied)

## Result
**FAILED - Correctness test failed**

All tests failed with significant mismatches (16,875+ elements per test case).
The reference implementation uses `doweight_stage1=False`, meaning weights are applied
in stage 2. Changing this breaks correctness.

## Conclusion
`doweight_stage1` cannot be changed - it's not a tuning parameter but a correctness requirement.
