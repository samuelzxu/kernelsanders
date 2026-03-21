# Attempt 251: Split nope+rope dot_scaled with pre-quantized Q

## Key fix: split 576 dims into nope(512)+rope(64)
- nope: 256 packed bytes, 16 scale groups → ALL powers of 2!
- rope: 32 packed bytes, 2 scale groups → powers of 2!
- Avoids the 288 packed / 18 scales which aren't powers of 2

## Previous failures:
- 243: _mxfp4_quant_op timeout (17 min — was infra issue?)
- 248: KeyError float4_e2m1fn_x2 (need uint8 cast)
- 249: ValueError arange range not power of 2 (288, 18)
- 250: Reduction dim mismatch (padded 288→512 caused issues)
