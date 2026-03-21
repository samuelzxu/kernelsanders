# #128 Disable LLVM LSR and Machine LICM Passes

## Inspiration
Leaderboard: `submission_v220_disable_lsr_disable_machine_licm.py` at 9.7µs!
This is 6µs faster than our 15.9µs. The technique:
- LSR (Loop Strength Reduction): optimizes loop induction variables but
  increases register pressure → register spills → slower
- Machine LICM (Loop Invariant Code Motion): hoists computations out of loops
  but increases live register count → spills for register-heavy FP4 GEMM

## Implementation
```python
os.environ['DISABLE_LLVM_OPT'] = 'disable-lsr,disable-machine-licm'
```
Set before any Triton import to affect JIT compilation.

## Expected Impact
Could be massive (up to 6µs improvement) if register pressure is the bottleneck.
The FP4 MFMA instruction uses 16 fp32 accumulators + 8 uint32 A registers +
8 uint32 B registers per instruction. With BSM=32 BSN=64, that's 2048 fp32
accumulators = significant register pressure.
