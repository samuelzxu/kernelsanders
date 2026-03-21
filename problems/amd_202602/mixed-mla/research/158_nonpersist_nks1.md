# Attempt 158: Non-persistent nks=1 fast path for bs=256/kv<=1024

## Key Discovery from aiter/mla.py Source
When num_kv_splits=1 and Q is FP8, stage1 writes directly to the output
buffer (logits = o.view(...)). No reduce step, no lg/ls buffers.

## Code Path (from mla.py lines 205-257)
```
logits = o.view((total_s, 1, nhead, v_head_dim))  # bf16 view of output
aiter.mla_decode_stage1_asm_fwd(q_fp8, kv_fp8, ..., nksi, None, None, None, ...)
return logits.view(total_s, nhead, v_head_dim)  # = o reshaped
```

## Savings for bs=256/kv=1024
- Skip 6 metadata allocations: ~2µs
- Skip get_mla_metadata_v1: ~3µs
- Skip lg/ls allocation: ~2µs
- Skip mla_reduce_v1: ~5µs
- Add Q quantization: +3µs
- Net saving: ~9µs → 113µs → ~104µs

## Why only bs>=256
- bs=256: 256 CTAs with nks=1 = 1 full wave on MI355X ✓
- bs=64: 64 CTAs with nks=1 = 0.25 wave = bad utilization ✗
