# v58: Minimal Prebuild with Triton Quant Override

## Key Innovation
1. Monkey-patch `get_quant` in fused_moe to use Triton quant for per_1x32
   → Avoids triggering module_quant build (~25s saved)
2. Trigger module_moe_asm build on first custom_kernel call (~30s)
3. Force 1-stage for inter_dim <= 1024

## Expected JIT Build Timeline (Cold Runner)
- module_moe_sorting: 25s (needed by both paths)
- module_moe_ck2stages: 105s (needed by 2-stage for E=257/d=2048)
- module_moe_asm: 30s (triggered by our prebuild)
- module_quant: SKIPPED (using Triton quant instead)
- Total: 160s ≈ 2.7 min

## Total Time Budget
- JIT builds: ~160s
- Test phase: ~30s
- Benchmark warmup: ~5s
- 7 benchmarks: ~5min (1-stage for 6 shapes, 2-stage for 1)
- Total: ~8.5 min (under 12 min!)
