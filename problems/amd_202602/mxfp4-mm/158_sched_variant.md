# #158 - AMD instruction_sched_variant="default" + O1

## Hypothesis

The Triton AMD backend has an underdocumented parameter `instruction_sched_variant`
in `HIPOptions` (triton/third_party/amd/backend/compiler.py):

```python
@dataclass(frozen=True)
class HIPOptions:
    instruction_sched_variant: str = "none"
    ...
```

When set to `"default"`, the compiler runs the
`amd.passes.ttgpuir.insert_instruction_sched_hints` pass which inserts
AMD-specific scheduling metadata into the IR. This tells the hardware
scheduler how to interleave MFMA instructions with memory loads for
better execution unit utilization.

The default is `"none"` (disabled). This experiment patches the default
to `"default"` alongside the existing O1 optimization, then measures
whether the scheduling hints improve the FP4 GEMM throughput.

## Implementation

Patches `HIPOptions.instruction_sched_variant` default from `"none"` to
`"default"` in the AMD compiler module (same approach as O1 patch). Combined
with the O1 LLVM optimization for consistency with best known baseline.

## Changes from #148

- Added patch: `instruction_sched_variant: str = "none"` → `"default"`
- All configs identical to #148 (isolating single variable)
