# Attempt 179: V slice inside compiled fn + fullgraph=True

## Changes vs 170
1. V slice (kv[:,:,:DV]) moved inside compiled function
   - Inductor can fuse slice with bmm (avoid non-contiguous tensor materialization)
2. fullgraph=True: forces single graph, max optimization
   - Never tried before — could enable deeper kernel fusion
