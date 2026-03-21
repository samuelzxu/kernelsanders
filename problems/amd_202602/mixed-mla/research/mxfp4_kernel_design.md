# MXFP4 Flash-Decode Kernel Design (HIP C++ via load_inline)

## Goal
Write a fused MLA decode kernel that loads MXFP4 KV cache and dequantizes
in registers, avoiding the 2x bandwidth overhead of fp8/bf16 KV.

## MLA Decode Operation
For each (batch, head):
  scores[t] = Q[batch, head, :576] · K[batch, t, :576] * sm_scale
  attn = softmax(scores)
  out[batch, head, :512] = Σ_t attn[t] * V[batch, t, :512]

## MXFP4 Data Layout
- KV buffer: fp4x2 packed, shape (total_kv, 1, 288) - each byte holds 2 fp4 values
- Scales: fp8_e8m0, shape (total_kv, N_blocks) where N_blocks = 576/32 = 18
- To dequantize block b of token t:
  1. Load 16 bytes of fp4x2 data (32 fp4 values)
  2. Load 1 byte of e8m0 scale
  3. Unpack fp4 → float using lookup table
  4. Multiply by 2^(scale - 127)

## Kernel Architecture
- Grid: (batch * num_heads * num_splits,)
- Each thread block handles one (batch, head, split) triple
- Thread block size: 256 threads
- Each thread processes BLOCK_N KV tokens

### Per thread block:
1. Load Q vector (576 floats) into shared memory
2. For each chunk of BLOCK_N KV tokens in this split:
   a. Load MXFP4 K data (288 bytes * BLOCK_N) + scales (18 * BLOCK_N)
   b. Dequantize K to fp32 in registers (lookup table + scale multiply)
   c. Compute QK^T dot products (576-dim reduction per token)
   d. Online softmax update
   e. Load MXFP4 V data (256 bytes * BLOCK_N) + scales (16 * BLOCK_N)
   f. Dequantize V to fp32
   g. Accumulate weighted V
3. Store partial output + LSE

## MXFP4 Dequantization in HIP
```cpp
__device__ float mxfp4_to_float(uint8_t packed, int idx) {
    static const float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    uint8_t nibble = (idx == 0) ? (packed & 0xF) : (packed >> 4);
    return LUT[nibble];
}

__device__ float e8m0_to_float(uint8_t scale) {
    // 2^(scale - 127)
    int exp = (int)scale;
    uint32_t f32_bits = exp << 23;
    return __int_as_float(f32_bits);
}
```

## Bandwidth Analysis
For bs=256, kv=8192:
- MXFP4 KV: 256 * 8192 * (288 + 18) bytes = 642 MB
- fp8 KV:   256 * 8192 * 576 bytes = 1.2 GB
- bf16 KV:  256 * 8192 * 1152 bytes = 2.4 GB
- Savings: 1.87x over fp8, 3.74x over bf16
