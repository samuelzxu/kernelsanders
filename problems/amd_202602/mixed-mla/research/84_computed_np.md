# Attempt 84: Computed np_ directly

## Changes
- Compute np_ = bs * nks instead of rpm.size(0)
- Use bs instead of q.shape[0] for output allocation
- Both avoid accessing GPU tensor metadata

## Results
~76 µs benchmark geomean (within noise of attempt 75's ~77 µs).
Minor: rpm.size(0) was already a fast CPU call.

## Current legitimate best: ~76-77 µs benchmark geomean
84 attempts total. No more obvious optimization paths available.
