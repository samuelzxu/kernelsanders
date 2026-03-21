# Attempt 168: Scores allocation inside compiled fn + kvg=32 for kv<=1024

## Two changes
1. Move scores_buf allocation inside compiled function (torch.compile
   can optimize internal allocations via memory pool reuse)
2. kvg=32 for kv<=1024 a16w8 path (isolated test, never done alone before)
