# MXFP4-MM Kernel Optimization Research Flow

## Objective
Continuously iterate and improve the submission.py kernel for the MXFP4 matrix multiplication problem on AMD MI355X GPU.

## Submission Command
```bash
popcorn-cli submit --leaderboard amd-mxfp4-mm --mode leaderboard --gpu MI355X submission.py --no-tui
```

Run this in the background and check output every 90 seconds for completion.

## Research Flow

### 1. Initial Setup
- Create a `research/` directory to track attempts
- Name files as `{i}_{shorthand}.md` (e.g., `1_baseline.md`, `2_doweight.md`)
- Read current `submission.py` and `reference.py` to understand the baseline

### 2. For Each Attempt
1. **Document the hypothesis** in `research/{i}_{name}.md` BEFORE submitting
2. **Make minimal changes** to submission.py
3. **Submit and wait** for results (~3-5 minutes typically)
4. **Record results** with timing data in the research doc
5. **Analyze** what worked/didn't work
6. **Iterate** based on findings

### 3. Key Files to Examine
- `submission.py` - Your kernel implementation
- `reference.py` - Reference implementation and input generation
- `task.py` - Input/output type definitions
- `README.md` - Problem description and optimization hints

### 4. AITER Library Functions (for MXFP4)
- `aiter.gemm_a4w4()` - FP4 x FP4 GEMM with per-block scaling
- `aiter.ops.triton.quant.dynamic_mxfp4_quant()` - Dynamic MXFP4 quantization
- `aiter.ops.shuffle.shuffle_weight()` - Weight shuffling for CK kernels
- `aiter.utility.fp4_utils.e8m0_shuffle()` - Scale shuffling

### 5. Optimization Ideas for MXFP4-MM
1. **Tile size tuning** - Different M/N/K may benefit from different tiles
2. **Memory coalescing** - Ensure optimal memory access patterns
3. **Quantization fusion** - Fuse quant into GEMM prologue
4. **Split-K parallelism** - For large K dimensions
5. **Non-temporal loads** - For large matrices that don't fit in cache

### 6. Checking Submission Progress
```bash
# Check output file
cat /private/tmp/claude/.../tasks/{task_id}.output | tail -50

# Wait 90 seconds then check
sleep 90 && cat /private/tmp/claude/.../tasks/{task_id}.output | tail -50
```

### 7. Result Analysis Template
```markdown
## Result (Ranked Benchmark)
| Shape | Time [µs] | Reference [µs] | Speedup |
|-------|-----------|----------------|---------|
| ...   | ...       | ...            | ...     |

**Geometric Mean Speedup: X.XXx**
```

## Important Notes
- The leaderboard uses geometric mean of all benchmark cases
- Correctness is checked with rtol=1e-2, atol=1e-2
- MXFP4 uses 32-element blocks for scaling (per_1x32)
- Weight shuffling layout is (16, 16) for CK kernels
- Dimensions must be padded to 256-alignment

## Quick Start
1. Read submission.py and reference.py
2. Create research/1_baseline.md documenting current approach
3. Submit baseline: `popcorn-cli submit --leaderboard amd-mxfp4-mm --mode leaderboard --gpu MI355X submission.py --no-tui`
4. Wait for results and iterate
