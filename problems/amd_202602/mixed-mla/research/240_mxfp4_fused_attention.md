# Attempt 240: Fused MXFP4 Decode Attention Kernel

## The Breakthrough Discovery
Top leaderboard: 13.5µs. Our score: 43.8µs (leaderboard, different from local 70µs).
The gap is from using fp8 KV (576 bytes/token) vs MXFP4 KV (306 bytes/token).
"5µs overhead + MXFP4 bandwidth" model → 13.03µs geomean. Matches top score exactly.

## Design
Custom Triton kernel based on mla_decode_rope.py stage1, modified to:
1. Load MXFP4 KV (fp4x2 packed + e8m0 scales) instead of bf16
2. Dequant in registers using tl operations (no HBM write)
3. Split 576-dim into nope (512) + rope (64) for separate dot products
4. Online softmax with split-K parallelization
5. Variable-length batching via kv_indptr

## Key Triton Operations
- `tl.load` fp4x2 data (half the bytes of bf16)
- Block scale dequant: fp4_value * 2^(e8m0_scale - 127)
- `tl.dot` for attention score computation
- Online softmax accumulation

## Expected Impact
- 2x bandwidth savings for KV reads (288 bytes vs 576 bytes per token)
- Eliminate metadata overhead (no get_mla_metadata_v1)
- Single kernel (no separate reduce for small configs)
- Target: ~15-20µs geomean (from ~44µs current leaderboard score)
