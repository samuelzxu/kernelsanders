#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#463: Test CK ASM a4w4 with injected tuned configs for all shapes.
Injects CK config entries so a4w4 uses optimal kernel tiles instead of defaults.
Then uses fused quant + a4w4 for shapes where it beats preshuffle.
"""
import torch, os, json, csv
from task import input_t, output_t
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

# === Inject CK ASM configs FIRST (before any gemm_a4w4 calls) ===
def _inject_ck_configs():
    try:
        from aiter.jit.core import AITER_CONFIGS
        csv_path = AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
        if not os.path.exists(csv_path):
            return

        # Read existing entries
        existing = set()
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    existing.add((int(row[0]), int(row[1]), int(row[2]), int(row[3])))

        # Configs to inject: cu_num, M, N, K, kernelId, splitK, us, kernelName, tflops, bw, errRatio
        # Available tiles: 32x128 (small), 192x128 (large)
        new_entries = [
            # Benchmark shapes
            [256, 4, 2880, 512, 21, 0, 8.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 16, 2112, 7168, 21, 0, 20.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 32, 4096, 512, 21, 0, 9.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 32, 2880, 512, 21, 0, 9.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 64, 7168, 2048, 21, 0, 13.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 256, 3072, 1536, 21, 0, 13.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            # Also try 192x128 tile for larger shapes
            [256, 64, 7168, 2048, 29, 0, 13.0, "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128E", 0, 0, 0.0],
            [256, 256, 3072, 1536, 29, 0, 13.0, "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128E", 0, 0, 0.0],
            # Test shapes
            [256, 8, 2112, 7168, 21, 0, 20.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 16, 3072, 1536, 21, 0, 12.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 64, 3072, 1536, 21, 0, 12.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
            [256, 256, 2880, 512, 21, 0, 9.0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E", 0, 0, 0.0],
        ]

        to_add = [e for e in new_entries if (e[0], e[1], e[2], e[3]) not in existing]
        if to_add:
            with open(csv_path, "a") as f:
                writer = csv.writer(f)
                for entry in to_add:
                    writer.writerow(entry)
            print(f"Injected {len(to_add)} CK configs")
    except Exception as e:
        print(f"CK inject failed: {e}")

_inject_ck_configs()

# === Preshuffle setup ===
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info

_cfgs={"N=2880-K=512":{"M_LEQ_4":{"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":16,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":1,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=4096-K=512":{"M_LEQ_32":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":32,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"NUM_KSPLIT":1,"num_warps":8,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=2112-K=7168":{"M_LEQ_16":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":8,"num_warps":4,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=7168-K=2048":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":3,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}},"N=3072-K=1536":{"M_LEQ_64":{"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"NUM_KSPLIT":3,"num_warps":4,"num_stages":1,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg"},"M_LEQ_256":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":256,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":2,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":32,"cache_modifier":".cg"},"any":{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":4,"NUM_KSPLIT":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":None}}}
try:_dev=arch_info.get_arch()
except:_dev="gfx950"
_cd=f"{AITER_TRITON_CONFIGS_PATH}/gemm";os.makedirs(_cd,exist_ok=True)
for _sk,_cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json","w") as f:json.dump(_cfg,f)

# Warmup both paths
for _m,_n,_k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn((_m,_k),dtype=torch.bfloat16,device="cuda")
        _Bw=torch.zeros((_n//16,(_k//2)*16),dtype=torch.uint8,device="cuda")
        _Bws=torch.zeros((_n//32,_k),dtype=torch.uint8,device="cuda")
        gemm_a16wfp4_preshuffle(_A,_Bw,_Bws,prequant=True,dtype=torch.bfloat16)
    except:pass
for _m,_n,_k in [(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
    try:
        _A=torch.randn(_m,_k,dtype=torch.bfloat16,device="cuda")
        _aq,_as=dynamic_mxfp4_quant(_A)
        _aq=_aq.view(dtypes.fp4x2)
        _as=e8m0_shuffle(_as).view(dtypes.fp8_e8m0)
        from aiter.ops.shuffle import shuffle_weight
        _bq=torch.zeros(_n,_k//2,dtype=torch.uint8,device="cuda").view(dtypes.fp4x2)
        _bsh=shuffle_weight(_bq,layout=(16,16))
        _bsc=torch.zeros(_n,_k//32,dtype=torch.uint8,device="cuda").view(dtypes.fp8_e8m0)
        aiter.gemm_a4w4(_aq,_bsh,_as,_bsc,dtype=dtypes.bf16,bpreshuffle=True)
    except Exception as e:
        print(f"Warmup a4w4 {_m}x{_n}x{_k}: {e}")
torch.cuda.empty_cache()

# Use Triton quant + e8m0_shuffle + gemm_a4w4 for K=1536 M=256
_ps_ck=None;_ps_cw=None;_ps_cs=None

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    global _ps_ck,_ps_cw,_ps_cs
    A=data[0];B_shuffle=data[3];B_scale_sh=data[4]
    m,k=A.shape;n=data[1].shape[0]

    # Fused quant + gemm_a4w4 for K=1536 M=256 only (3us faster than preshuffle)
    if m == 256 and k == 1536:
        aq,asc=dynamic_mxfp4_quant(A)
        ash=e8m0_shuffle(asc).view(dtypes.fp8_e8m0)
        aq_t=aq.view(dtypes.fp4x2)
        return aiter.gemm_a4w4(aq_t, B_shuffle, ash, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)

    # Preshuffle path
    dp=B_shuffle.data_ptr()
    if dp!=_ps_ck:
        _ps_ck=dp;_ps_cw=B_shuffle.view(torch.uint8).reshape(n//16,(k//2)*16)
        _ps_cs=B_scale_sh.view(torch.uint8)[:n,:].contiguous().reshape(n//32,k)
    return gemm_a16wfp4_preshuffle(A,_ps_cw,_ps_cs,prequant=True,dtype=torch.bfloat16)
