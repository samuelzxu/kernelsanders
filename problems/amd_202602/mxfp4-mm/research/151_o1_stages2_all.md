# #151 O1 + num_stages=2 for all shapes
M=64: 20.4µs (worse than #148's 19.6µs). Pipeline depth (num_stages=4)
is still needed for K=2048 even with O1. The O3-tuned num_stages values
remain optimal with O1 codegen.
