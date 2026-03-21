# Attempt 157: reduce-overhead torch.compile mode

## Change vs 156
- torch.compile mode: "default" → "reduce-overhead"
- Uses HIP graphs to batch kernel launches
- dynamic=False (default): separate graph per shape (bs=4, bs=32)
- Combined with TunableOp (same as 156)

## Expected Impact
- kv=1024 GEMM: potentially faster (HIP graph replay vs kernel dispatch)
- bs=4/kv=1024: 21.6 → ~18-19µs? (graph replay eliminates dispatch overhead)
- bs=32/kv=1024: 34.6 → ~30-32µs?
- Risk: HIP graph capture timeout or incompatibility
