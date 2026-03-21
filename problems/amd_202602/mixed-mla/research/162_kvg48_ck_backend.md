# Attempt 162: kvg=48 + CK-only GEMM backend

## kvg=48 for kv>1024
- Never tested before, but supported (in aiter's get_block_n_fp8 table)
- kv=8192/kvg=48: 170.67 groups... wait, 8192/48 = 170.67!
- THIS DOESN'T DIVIDE EVENLY. kvg must divide kv_seq_len.
- 8192 % 48 = 8192 - 170*48 = 8192 - 8160 = 32. NOT zero!
- WAIT: does kvg need to divide kv_seq_len? Or is it just a granularity hint?
- Need to check if this causes issues.
