# #138 LLVM_ARGS Environment Variable
No effect. The LLVM library linked into libtriton.so doesn't read LLVM_ARGS.
Also tried AMD_COMGR_ACTION_OPTIONS - no effect.

## Complete LLVM pass disabling attempts (128-138):
| # | Method | Result |
|---|--------|--------|
| 128 | DISABLE_LLVM_OPT=comma-flags | No effect |
| 129 | Same + cache clear | No effect |
| 130 | DISABLE_LLVM_OPT=1 | 2x slower (works!) |
| 131 | Space-separated flags | No effect |
| 132 | Just disable-lsr | No effect |
| 133 | Monkey-patch C extension | Wrong import path |
| 134 | File glob patch | Wrong patterns |
| 135 | Import-based file patch | PermissionError (read-only) |
| 136 | Copy+patch+redirect | Patch applied but flags param doesn't control passes |
| 137 | BSN=64 KSPLIT=14 (leaderboard) | +4.5µs worse (reduction overhead) |
| 138 | LLVM_ARGS env | No effect |

The specific-pass-disabling approach is BLOCKED on this runner.
