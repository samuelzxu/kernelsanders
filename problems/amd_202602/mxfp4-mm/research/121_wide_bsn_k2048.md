# #121 BSM=16 BSN=128 for K=2048

## Results
M=64: 20.7µs (worse than #118's 19.8µs)
BSM=16 creates narrower M-tiles → more tile setup overhead.
BSM=32 BSN=64 is optimal: 2 stacked 16-row MFMA tiles keeps CU busier.
