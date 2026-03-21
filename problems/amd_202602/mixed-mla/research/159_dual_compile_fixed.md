# Attempt 159: Dual compiled GEMM functions

## Problem
Attempt 152: single compiled function with dynamic=True helped kv=1024 but hurt kv=8192.
Attempt 156: kept kv=8192 uncompiled — left 42.5µs on the table.

## Solution
Two separate compiled functions, each specialized:
- _compiled_short (dynamic=True): handles bs=4 and bs=32 with kv=1024
- _compiled_long (dynamic=False): handles only bs=4 with kv=8192

## Expected
- kv=1024: same as 156 (21.6, 34.6µs)
- kv=8192 GEMM: potentially improved from 42.5µs → ~38µs if fixed-shape compilation
  produces a better kernel than the general hipBLAS dispatch
