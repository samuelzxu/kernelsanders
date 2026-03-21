# #118 num_stages=4 for K=2048 Only - NEW BEST

## Change
K=2048 M_LEQ_64: num_stages=2→4 (deeper pipeline for 4 K-iterations)
K=7168: keep num_stages=3 (#117 showed worse with num_stages=4)

## Results (Ranked) - NEW BEST ~15.9µs geomean
- M=4: 12.1µs (-0.3µs, possibly noise)
- M=16: 21.1µs (same)
- M=32: 12.6µs (same)
- M=64: 19.8µs (-0.4µs ✓ consistent with #117)
- M=256: 19.8µs (-0.2µs)

## Analysis
num_stages=4 enables 4-stage software pipeline for the K-dimension loop.
With BSK=512 and K=2048: 4 K-iterations = exactly one load-compute overlap
per pipeline stage. This maximizes memory-compute overlap.

For K=7168 (7 iterations after get_splitk), num_stages=3 is already sufficient.
Adding a 4th stage increases register pressure without benefit (7 iters already
provides enough overlap with 3 stages).
