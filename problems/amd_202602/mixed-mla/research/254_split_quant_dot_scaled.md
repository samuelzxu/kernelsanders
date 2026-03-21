# Attempt 254: Split nope+rope with _mxfp4_quant_op inside kernel

## Key insight: _mxfp4_quant_op MUST be used inside the kernel
tl.dot_scaled requires scales in MFMA-compatible layout. Only _mxfp4_quant_op
produces this layout. Loading pre-computed scales with tl.load doesn't work.

## Split dimensions (all power of 2):
- Nope Q: bf16 (32, 512) → _mxfp4_quant_op → e2m1 (32, 256) + scales (32, 16)
- Rope Q: bf16 (32, 64) → _mxfp4_quant_op → e2m1 (32, 32) + scales (32, 2→4 padded)
- K nope: load MXFP4 (32, 256) + scales (32, 16) from kv_data["mxfp4"]
- K rope: load MXFP4 (32, 32) + scales (32, 2→4 padded)
- V: bf16 (32, 512) from kv_data["bf16"]

## Previous failures:
- 248: KeyError float4_e2m1fn_x2 → need .view(torch.uint8) on kv_fp4
- 249: ValueError arange not power of 2 → 288, 18 aren't powers of 2
- 250/251/252/253: Reduction dim assertion → scales from tl.load not in MFMA layout
- 243: "17 min timeout" → was infra issue, not compilation time!
