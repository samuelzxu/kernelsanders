# Attempt 175: Non-persistent mla_decode_fwd for kv<=1024

## Hypothesis
Non-persistent mode skips ALL metadata overhead (~6µs):
- No get_mla_metadata_info_v1
- No 6 buffer allocations
- No get_mla_metadata_v1 GPU kernel

## Risks
- Triton stage2 reduce kernel needs JIT compilation (first call)
- mla_decode_fwd Python wrapper adds ~2µs overhead
- Non-persistent mode uses bf16 Q (a16w8) — matches current kv<=1024 path
- Correctness: attempt 158 failed with non-persistent nks=1, but nks=8 might work
