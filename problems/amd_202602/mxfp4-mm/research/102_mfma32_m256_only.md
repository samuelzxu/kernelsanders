# #102 MFMA 32x32 for M=256 Only

## Changes
Only change from #92: M_LEQ_256 tier for K=1536 uses matrix_instr_nonkdim=32.
All other shapes keep matrix_instr_nonkdim=16.

## Results (Ranked) - NEW BEST
- M=4: 12.2µs (same)
- M=16: 21.3µs (same)
- M=32, N=4096: 12.6µs (same)
- M=32, N=2880: 12.5µs (same)
- M=64: 20.0µs (-0.5µs, might be noise since config unchanged)
- M=256: 19.5µs (-1.1µs ✓ confirmed improvement from MFMA 32x32)
- Geomean: ~15.9µs (from 16.1µs)

## Analysis
MFMA 32x32 is better for M=256 (BSM=64): the 32x32 MFMA tile better matches
the BSM=64 tile size (two 32x32 tiles vs four 16x16 tiles). Higher instruction
throughput with fewer MFMA dispatches.

For M<=64 shapes, MFMA 16x16 remains better due to smaller tile sizes.

CURRENT BEST: #102 at ~15.9µs ranked geomean.
