# #133-135 LLVM Pass Patching Attempts

## Findings
- #133: Monkey-patch _triton.llvm.optimize_module → wrong import path
- #134: File glob pattern → wrong paths
- #135: Found correct path but **PermissionError**: `/usr/local/lib/python3.12/dist-packages/triton/backends/amd/compiler.py` is READ-ONLY

## Key info from #135 stderr
The Triton AMD backend is at:
`/usr/local/lib/python3.12/dist-packages/triton/backends/amd/compiler.py`
But it's owned by root and not writable by the runner user.

## Next approach
Copy compiler.py to writable location, patch it, redirect Python's import.
Or: use `importlib` to load a patched version and replace the module in sys.modules.
