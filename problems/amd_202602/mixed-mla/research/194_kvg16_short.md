# Attempt 194: Revert kvg=16 for kv<=1024 (fix correctness failures)

## CRITICAL BUG FOUND
ALL recent correctness failures at bs=64/kv=1024 used kvg=32:
- 187c: batch 44, dim 363 (16 elements)
- 190: batch 40, dims 353/500 (64 elements)
- 193: batch 4, dim 502 (16 elements)

Before attempt 160 (kvg=16 for kv<=1024), there were ZERO failures
across 130+ submissions at bs=64/kv=1024.

## Fix
Revert kv<=1024 assembly to kvg=16 (proven reliable).
Keep kvg=32 for kv>1024 (no failures observed).
