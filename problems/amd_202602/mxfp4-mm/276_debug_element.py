#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
#276: Debug ONE element. Compute output[0][0] with the HIP kernel AND preshuffle,
compare intermediate values (A_fp4 bytes, B bytes, scales).
"""
import os, shutil, sys, json, torch
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_cfgs = {"N=2880-K=512": {"M_LEQ_4": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=4096-K=512": {"M_LEQ_32": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=7168-K=2048": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 512, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 2, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}, "N=3072-K=1536": {"M_LEQ_64": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 3, "num_warps": 4, "num_stages": 1, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "M_LEQ_256": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 3, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}, "any": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": None}}}
try: _dev = arch_info.get_arch()
except: _dev = "gfx950"
_cd = f"{AITER_TRITON_CONFIGS_PATH}/gemm"
os.makedirs(_cd, exist_ok=True)
for _sk, _cfg in _cfgs.items():
    with open(f"{_cd}/{_dev}-GEMM-A16WFP4_PRESHUFFLED-{_sk}.json", "w") as f:
        json.dump(_cfg, f)

_bc = [None, None, None]
_first = [True]

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    dp = B_shuffle.data_ptr()
    if dp != _bc[0]:
        _bc[0] = dp
        _bc[1] = B_shuffle.view(torch.uint8).reshape(n // 16, (k // 2) * 16)
        _bc[2] = B_scale_sh.view(torch.uint8)[:n, :].contiguous().reshape(n // 32, k)

    if _first[0]:
        _first[0] = False
        # Compare A quantization: my sw_quant vs dynamic_mxfp4_quant
        A_q_ref, A_scale_ref = dynamic_mxfp4_quant(A)
        A_q_ref_u8 = A_q_ref.view(torch.uint8)
        A_scale_ref_u8 = A_scale_ref.view(torch.uint8)[:m, :k//32].contiguous()

        # Check B_q bytes vs what preshuffle kernel reads from B_w
        B_q_u8 = B_q.view(torch.uint8)
        B_w = _bc[1]

        # For output[0][0]: row 0, col 0
        # A row 0: A_q_ref_u8[0, :8] (first 8 bytes = 16 FP4 values)
        # B col 0 (N-row 0): B_q_u8[0, :8]
        # B_w for N-row 0: super_row=0, n_within=0
        # B_w[0, kb*512 + kh*256 + 0*16 + ki]
        print(f"[276] A_q[0,:8] = {A_q_ref_u8[0,:8].tolist()}", file=sys.stderr)
        print(f"[276] A_scale[0,:4] = {A_scale_ref_u8[0,:4].tolist()}", file=sys.stderr)
        print(f"[276] B_q[0,:8] = {B_q_u8[0,:8].tolist()}", file=sys.stderr)

        # Read B_w for N-row 0: should give same bytes as B_q[0,:]
        bw_bytes_row0 = []
        for kbyte in range(min(8, k//2)):
            kb = kbyte // 32
            kh = (kbyte % 32) // 16
            ki = kbyte % 16
            bw_idx = kb * 512 + kh * 256 + 0 * 16 + ki
            if bw_idx < B_w.shape[1]:
                bw_bytes_row0.append(B_w[0, bw_idx].item())
            else:
                bw_bytes_row0.append(-1)
        print(f"[276] B_w->row0[:8] = {bw_bytes_row0[:8]}", file=sys.stderr)
        print(f"[276] Match B_q==B_w: {B_q_u8[0,:8].tolist() == bw_bytes_row0[:8]}", file=sys.stderr)

        # B_scale: unshuffle and check
        def _unshuffle(ssh, nn, kk):
            s = ssh.view(torch.uint8)
            sm, sn = s.shape
            return s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm, sn)[:nn, :kk//32].contiguous()
        B_scale_unsh = _unshuffle(B_scale_sh, n, k)
        _, B_scale_fresh = dynamic_mxfp4_quant(B)
        B_scale_fresh_u8 = B_scale_fresh.view(torch.uint8)[:n, :k//32].contiguous()
        print(f"[276] B_scale_unsh[0,:4] = {B_scale_unsh[0,:4].tolist()}", file=sys.stderr)
        print(f"[276] B_scale_fresh[0,:4] = {B_scale_fresh_u8[0,:4].tolist()}", file=sys.stderr)

        # A_scale strides check
        print(f"[276] A_scale_ref strides={A_scale_ref.stride()} shape={A_scale_ref.shape}", file=sys.stderr)
        print(f"[276] A_scale_ref contig strides={A_scale_ref_u8.stride()}", file=sys.stderr)

    return gemm_a16wfp4_preshuffle(A, _bc[1], _bc[2], prequant=True, dtype=torch.bfloat16)
