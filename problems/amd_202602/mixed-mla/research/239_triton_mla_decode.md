# Attempt 239: Triton MLA decode kernel (completely different approach!)

## Discovery
aiter has a Triton MLA decode kernel (`mla_decode_rope.py`) purpose-built for
DeepSeek's 512+64 dim split. Pre-configured for MI355X (gfx950).

## Key Differences from Assembly Path
- Handles 512 (nope) + 64 (rope) dims SEPARATELY (more efficient than generic 576)
- No metadata computation (no get_mla_metadata_v1)
- Built-in two-stage split-K reduction (no separate reduce call)
- Uses bf16 KV directly (reads kv_data["bf16"])
- Triton auto-tuned tile sizes (BLOCK_N=32, BLOCK_H=16)

## Risks
- Triton JIT compilation might timeout on MI355X
- bf16 KV = 2x more bandwidth than fp8 (1152 vs 576 bytes per token)
- The attn_logits buffer shape might be wrong
