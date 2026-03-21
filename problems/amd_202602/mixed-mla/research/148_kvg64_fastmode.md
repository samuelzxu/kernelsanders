# Attempt 148: kvg=64 for kv<=1024 + fast_mode=True

## Changes vs 137
1. kvg=64 for kv<=1024 (was 16): 16 groups vs 64 groups per batch item
   - Fewer groups = less metadata HBM overhead
   - Validated: kvg=64 > kvg=16 for kv>1024 (attempt 145)
   - Risk: 2 groups per CTA with nks=8 (1024/8/64=2) might be too few

2. fast_mode=True for get_mla_metadata_info_v1 and get_mla_metadata_v1
   - Default is True in aiter, we were using False
   - Might compute metadata faster with some approximation
   - Risk: might produce incorrect metadata → wrong results
