# #126 Use Repo's Original Config for N=2112-K=7168

## Discovery
We've been OVERRIDING the repo's tuned config with our own:
- Our config: BSN=64, KSPLIT=8(→7), num_stages=3 = 231 tiles (0.9 waves)
- Repo config: BSN=32, KSPLIT=14, num_stages=2 = 924 tiles (3.6 waves)

The repo config was tuned by AMD engineers for MI355X (256 CUs).
It has 4x more tiles and much better CU occupancy.

## Approach
Remove N=2112-K=7168 from our config injection → repo's config used.

## Results
M=16 K=7168: 29.3µs (MUCH WORSE, +8µs vs #118's 21µs!)
The 14-way reduction kernel overhead is enormous.
924 tiles with great occupancy doesn't help when reduction dominates.

## Conclusion
Repo's KSPLIT=14 was tuned for a different scenario (maybe CK kernel or
different Triton version). Our BSN=64 KSPLIT=8(→7) is far superior.
Validates our original config choice.
