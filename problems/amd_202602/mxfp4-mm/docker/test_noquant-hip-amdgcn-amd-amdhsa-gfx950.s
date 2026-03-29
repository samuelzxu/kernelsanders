	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	noquant                 ; -- Begin function noquant
	.globl	noquant
	.p2align	8
	.type	noquant,@function
noquant:                                ; @noquant
; %bb.0:
	s_load_dword s3, s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s3, 8
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	s_ashr_i32 s4, s2, 31
	s_lshr_b32 s4, s4, 29
	s_add_i32 s4, s2, s4
	s_ashr_i32 s5, s4, 3
	s_and_b32 s4, s4, -8
	s_add_i32 s6, s3, 7
	s_sub_i32 s4, s2, s4
	s_lshr_b32 s6, s6, 3
	s_mul_i32 s4, s6, s4
	s_add_i32 s4, s4, s5
	s_cmp_lt_i32 s4, s3
	s_cselect_b32 s2, s4, s2
.LBB0_2:
	s_load_dwordx4 s[4:7], s[0:1], 0x10
	s_abs_i32 s11, s2
	v_bfe_u32 v28, v0, 5, 1
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	s_waitcnt lgkmcnt(0)
	s_add_i32 s3, s4, 31
	s_add_i32 s8, s5, 31
	s_ashr_i32 s9, s3, 31
	s_ashr_i32 s10, s8, 31
	s_lshr_b32 s9, s9, 27
	s_lshr_b32 s10, s10, 27
	s_add_i32 s3, s3, s9
	s_add_i32 s8, s8, s10
	s_ashr_i32 s3, s3, 5
	s_ashr_i32 s8, s8, 5
	s_min_i32 s9, s3, 8
	s_mul_i32 s10, s9, s8
	s_abs_i32 s8, s10
	v_cvt_f32_u32_e32 v1, s8
	s_sub_i32 s13, 0, s8
	s_xor_b32 s12, s2, s10
	s_ashr_i32 s12, s12, 31
	v_rcp_iflag_f32_e32 v1, v1
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_readfirstlane_b32 s14, v1
	s_mul_i32 s13, s13, s14
	s_mul_hi_u32 s13, s14, s13
	s_add_i32 s14, s14, s13
	s_mul_hi_u32 s13, s11, s14
	s_mul_i32 s14, s13, s8
	s_sub_i32 s11, s11, s14
	s_add_i32 s15, s13, 1
	s_sub_i32 s14, s11, s8
	s_cmp_ge_u32 s11, s8
	s_cselect_b32 s13, s15, s13
	s_cselect_b32 s11, s14, s11
	s_add_i32 s14, s13, 1
	s_cmp_ge_u32 s11, s8
	s_cselect_b32 s8, s14, s13
	s_xor_b32 s8, s8, s12
	s_sub_i32 s11, s8, s12
	s_mul_i32 s14, s11, s9
	s_sub_i32 s3, s3, s14
	s_min_i32 s15, s3, s9
	s_abs_i32 s3, s15
	v_cvt_f32_u32_e32 v2, s3
	s_mul_i32 s11, s11, s10
	s_sub_i32 s16, s2, s11
	s_sub_i32 s10, 0, s3
	v_rcp_iflag_f32_e32 v2, v2
	s_abs_i32 s2, s16
	s_xor_b32 s9, s16, s15
	s_ashr_i32 s9, s9, 31
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_load_dwordx2 s[12:13], s[0:1], 0x8
	s_mov_b32 s8, 0
	v_and_b32_e32 v1, 31, v0
	v_readfirstlane_b32 s11, v2
	s_mul_i32 s10, s10, s11
	s_mul_hi_u32 s10, s11, s10
	s_add_i32 s11, s11, s10
	s_mul_hi_u32 s10, s2, s11
	s_mul_i32 s11, s10, s3
	s_sub_i32 s2, s2, s11
	s_add_i32 s17, s10, 1
	s_sub_i32 s11, s2, s3
	s_cmp_ge_u32 s2, s3
	s_cselect_b32 s10, s17, s10
	s_cselect_b32 s2, s11, s2
	s_add_i32 s11, s10, 1
	s_cmp_ge_u32 s2, s3
	s_cselect_b32 s2, s11, s10
	s_xor_b32 s2, s2, s9
	s_sub_i32 s18, s2, s9
	s_lshl_b32 s17, s18, 5
	s_cmpk_lt_i32 s6, 0x80
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	s_cbranch_scc1 .LBB0_9
; %bb.3:                                ; %.lr.ph
	s_ashr_i32 s2, s6, 31
	v_lshrrev_b32_e32 v2, 5, v0
	s_lshr_b32 s2, s2, 25
	v_lshrrev_b32_e32 v5, 6, v0
	s_add_i32 s2, s6, s2
	v_xor_b32_e32 v2, v2, v5
	s_ashr_i32 s19, s2, 7
	v_lshlrev_b32_e32 v3, 4, v0
	v_lshrrev_b32_e32 v6, 3, v0
	s_movk_i32 s2, 0x70
	v_lshlrev_b32_e32 v2, 4, v2
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_and_b32_e32 v29, 0x70, v3
	v_add_u32_e32 v4, s17, v6
	v_and_b32_e32 v2, 16, v2
	v_lshlrev_b32_e32 v5, 7, v6
	v_bitop3_b32 v3, v3, s2, v0 bitop3:0x48
	v_bitop3_b32 v30, v3, v5, v2 bitop3:0xde
	v_ashrrev_i32_e32 v2, 31, v4
	v_lshrrev_b32_e32 v2, 28, v2
	v_add_u32_e32 v2, v4, v2
	v_ashrrev_i32_e32 v5, 4, v2
	v_and_b32_e32 v2, 0xffffff0, v2
	v_cmp_gt_i32_e32 vcc, s5, v4
	v_sub_u32_e32 v4, v4, v2
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[0:1]
	v_mad_i64_i32 v[2:3], s[0:1], v5, s7, v[2:3]
	v_lshlrev_b32_e32 v5, 8, v0
	v_and_b32_e32 v18, 0x100, v5
	v_mov_b32_e32 v19, 0
	v_lshlrev_b32_e32 v4, 4, v4
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[18:19]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[26:27], v[2:3], 0, v[4:5]
	v_lshrrev_b32_e32 v4, 2, v0
	v_bfe_u32 v3, v0, 1, 3
	v_xor_b32_e32 v4, v4, v6
	v_lshlrev_b32_e32 v2, 4, v28
	v_bitop3_b32 v3, v4, v3, 1 bitop3:0x6c
	v_lshlrev_b32_e32 v3, 4, v3
	v_lshlrev_b32_e32 v4, 7, v1
	v_or_b32_e32 v5, 32, v2
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	v_bitop3_b32 v31, v2, v4, v3 bitop3:0xde
	v_bitop3_b32 v32, v5, v4, v3 bitop3:0xde
	v_or_b32_e32 v5, 64, v2
	v_or_b32_e32 v2, 0x60, v2
	v_and_b32_e32 v0, 7, v0
	s_mov_b32 s9, s8
	v_mov_b64_e32 v[24:25], s[10:11]
	v_bitop3_b32 v33, v5, v4, v3 bitop3:0xde
	v_bitop3_b32 v34, v2, v4, v3 bitop3:0xde
	v_lshlrev_b32_e32 v0, 8, v0
	s_xor_b64 s[0:1], vcc, -1
	v_mov_b32_e32 v35, 0x7f
	v_mov_b64_e32 v[22:23], s[8:9]
	v_mov_b32_e32 v2, v19
	v_mov_b32_e32 v3, v19
	v_mov_b32_e32 v4, v19
	v_mov_b32_e32 v5, v19
	v_mov_b32_e32 v6, v19
	v_mov_b32_e32 v7, v19
	v_mov_b32_e32 v8, v19
	v_mov_b32_e32 v9, v19
	v_mov_b32_e32 v10, v19
	v_mov_b32_e32 v11, v19
	v_mov_b32_e32 v12, v19
	v_mov_b32_e32 v13, v19
	v_mov_b32_e32 v14, v19
	v_mov_b32_e32 v15, v19
	v_mov_b32_e32 v16, v19
	v_mov_b32_e32 v17, v19
	s_branch .LBB0_5
.LBB0_4:                                ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[2:3]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_setprio 1
	ds_read_b128 v[36:39], v31
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_32x32x64_f8f6f4 v[2:17], v[22:25], v[36:39], v[2:17], v35, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[36:39], v32
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_32x32x64_f8f6f4 v[2:17], v[22:25], v[36:39], v[2:17], v35, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[36:39], v33
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_32x32x64_f8f6f4 v[2:17], v[22:25], v[36:39], v[2:17], v35, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[36:39], v34
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_32x32x64_f8f6f4 v[2:17], v[22:25], v[36:39], v[2:17], v35, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	s_setprio 0
	s_add_i32 s19, s19, -1
	v_add_u32_e32 v29, 0x80, v29
	s_cmp_eq_u32 s19, 0
	v_add_u32_e32 v0, 0x800, v0
	s_barrier
	s_cbranch_scc1 .LBB0_9
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	v_cmp_le_i32_e32 vcc, s6, v29
	s_or_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_xor_b64 s[2:3], exec, s[8:9]
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v18, v19
	v_mov_b32_e32 v20, v19
	v_mov_b32_e32 v21, v19
	ds_write_b128 v30, v[18:21]
; %bb.7:                                ; %Flow
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_andn2_saveexec_b64 s[2:3], s[2:3]
	s_cbranch_execz .LBB0_4
; %bb.8:                                ;   in Loop: Header=BB0_5 Depth=1
	v_and_b32_e32 v18, 0x7ffffe00, v0
	v_lshl_add_u64 v[20:21], v[26:27], 0, v[18:19]
	global_load_dwordx4 v[36:39], v[20:21], off
	s_waitcnt vmcnt(0)
	ds_write_b128 v30, v[36:39]
	s_branch .LBB0_4
.LBB0_9:                                ; %Flow177
	s_mul_i32 s18, s18, s15
	s_sub_i32 s0, s16, s18
	s_add_i32 s0, s0, s14
	v_lshlrev_b32_e32 v0, 2, v28
	v_lshl_or_b32 v18, s0, 5, v0
	v_or_b32_e32 v0, s17, v1
	v_cmp_gt_i32_e32 vcc, s5, v0
	v_cmp_gt_i32_e64 s[0:1], s4, v18
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_11
; %bb.10:
	v_bfe_u32 v1, v2, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v2, v1, s2
	v_mad_u64_u32 v[20:21], s[2:3], v18, s5, v[0:1]
	v_ashrrev_i32_e32 v21, 31, v20
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[12:13]
	global_store_short_d16_hi v[20:21], v1, off
.LBB0_11:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 1, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_13
; %bb.12:
	v_bfe_u32 v2, v3, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v19, v3, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v19, off
.LBB0_13:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 2, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_15
; %bb.14:
	v_bfe_u32 v2, v4, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v4, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_15:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 3, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_17
; %bb.16:
	v_bfe_u32 v2, v5, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v5, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_17:                               ; %.preheader.1
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 8, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_19
; %bb.18:
	v_bfe_u32 v2, v6, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v6, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_19:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 9, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_21
; %bb.20:
	v_bfe_u32 v2, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v7, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_21:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 10, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_23
; %bb.22:
	v_bfe_u32 v2, v8, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v8, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_23:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 11, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_bfe_u32 v2, v9, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v9, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_25:                               ; %.preheader.2
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 16, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_bfe_u32 v2, v10, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v10, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_27:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 17, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_bfe_u32 v2, v11, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v11, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_29:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 18, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_bfe_u32 v2, v12, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v12, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_31:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 19, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_bfe_u32 v2, v13, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v13, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_33:                               ; %.preheader.3
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 24, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_bfe_u32 v2, v14, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v14, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_35:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 25, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_bfe_u32 v2, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v15, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_37:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 26, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_39
; %bb.38:
	v_bfe_u32 v2, v16, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v16, v2, s2
	v_mad_u64_u32 v[2:3], s[2:3], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	global_store_short_d16_hi v[2:3], v4, off
.LBB0_39:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v1, 27, v18
	v_cmp_gt_i32_e64 s[0:1], s4, v1
	s_and_b64 s[0:1], s[0:1], vcc
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_41
; %bb.40:
	v_bfe_u32 v2, v17, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v2, v17, v2, s0
	v_mad_u64_u32 v[0:1], s[0:1], v1, s5, v[0:1]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[12:13]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_41:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel noquant
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 20
		.amdhsa_accum_offset 40
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	noquant, .Lfunc_end0-noquant
                                        ; -- End function
	.set noquant.num_vgpr, 40
	.set noquant.num_agpr, 0
	.set noquant.numbered_sgpr, 20
	.set noquant.private_seg_size, 0
	.set noquant.uses_vcc, 1
	.set noquant.uses_flat_scratch, 0
	.set noquant.has_dyn_sized_stack, 0
	.set noquant.has_recursion, 0
	.set noquant.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2388
; TotalNumSgprs: 26
; NumVgprs: 40
; NumAgprs: 0
; TotalNumVgprs: 40
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 26
; NumVGPRsForWavesPerEU: 40
; AccumOffset: 40
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_e07ec4e163847883,@object ; @__hip_cuid_e07ec4e163847883
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e07ec4e163847883
__hip_cuid_e07ec4e163847883:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e07ec4e163847883, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e07ec4e163847883
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           noquant
    .private_segment_fixed_size: 0
    .sgpr_count:     26
    .sgpr_spill_count: 0
    .symbol:         noquant.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     40
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
