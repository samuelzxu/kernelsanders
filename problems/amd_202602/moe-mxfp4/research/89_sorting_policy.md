# v89: Multi-Phase Sorting for E=33

## Change
dispatch_policy=2 (multi-phase) for E=33 shapes, dispatch_policy=0 (auto) for E=257.

## Hypothesis
Multi-phase sorting uses two kernels: one for histogram, one for sorting.
For E=33 (few experts), the multi-phase may be more efficient because
each phase has less work and can better utilize the GPU.
