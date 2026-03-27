	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	fp4_lds_opt             ; -- Begin function fp4_lds_opt
	.globl	fp4_lds_opt
	.p2align	8
	.type	fp4_lds_opt,@function
fp4_lds_opt:                            ; @fp4_lds_opt
; %bb.0:
	s_load_dwordx4 s[8:11], s[0:1], 0x28
	s_load_dword s3, s[0:1], 0x38
	s_ashr_i32 s6, s2, 31
	s_lshr_b32 s6, s6, 29
	s_add_i32 s6, s2, s6
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s4, s8, 31
	s_ashr_i32 s5, s9, 31
	s_lshr_b32 s4, s4, 27
	s_add_i32 s4, s8, s4
	s_lshr_b32 s5, s5, 27
	s_ashr_i32 s4, s4, 5
	s_add_i32 s5, s9, s5
	s_ashr_i32 s5, s5, 5
	s_min_i32 s11, s4, 8
	s_mul_i32 s5, s11, s5
	s_abs_i32 s12, s5
	v_cvt_f32_u32_e32 v1, s12
	s_ashr_i32 s7, s6, 3
	s_and_b32 s6, s6, -8
	s_sub_i32 s2, s2, s6
	v_rcp_iflag_f32_e32 v1, v1
	s_ashr_i32 s6, s3, 31
	s_lshr_b32 s6, s6, 29
	s_add_i32 s3, s3, s6
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s3, s3, 3
	s_mul_i32 s2, s3, s2
	s_add_i32 s2, s2, s7
	s_sub_i32 s7, 0, s12
	v_readfirstlane_b32 s13, v1
	s_mul_i32 s7, s7, s13
	s_mul_hi_u32 s7, s13, s7
	s_abs_i32 s6, s2
	s_add_i32 s13, s13, s7
	s_mul_hi_u32 s7, s6, s13
	s_mul_i32 s13, s7, s12
	s_xor_b32 s3, s2, s5
	s_sub_i32 s6, s6, s13
	s_ashr_i32 s3, s3, 31
	s_add_i32 s13, s7, 1
	s_sub_i32 s14, s6, s12
	s_cmp_ge_u32 s6, s12
	s_cselect_b32 s7, s13, s7
	s_cselect_b32 s6, s14, s6
	s_add_i32 s13, s7, 1
	s_cmp_ge_u32 s6, s12
	s_cselect_b32 s6, s13, s7
	s_xor_b32 s6, s6, s3
	s_sub_i32 s3, s6, s3
	s_mul_i32 s7, s3, s11
	s_sub_i32 s4, s4, s7
	s_min_i32 s4, s4, s11
	s_abs_i32 s6, s4
	v_cvt_f32_u32_e32 v1, s6
	s_mul_i32 s3, s3, s5
	s_sub_i32 s5, 0, s6
	s_sub_i32 s2, s2, s3
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s3, s2, s4
	s_ashr_i32 s22, s3, 31
	s_abs_i32 s3, s2
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_load_dwordx2 s[16:17], s[0:1], 0x0
	v_lshrrev_b32_e32 v17, 3, v0
	v_mov_b32_e32 v2, 0
	v_readfirstlane_b32 s11, v1
	s_mul_i32 s5, s5, s11
	s_mul_hi_u32 s5, s11, s5
	s_add_i32 s11, s11, s5
	s_mul_hi_u32 s5, s3, s11
	s_mul_i32 s11, s5, s6
	s_sub_i32 s3, s3, s11
	s_add_i32 s11, s5, 1
	s_sub_i32 s12, s3, s6
	s_cmp_ge_u32 s3, s6
	s_cselect_b32 s5, s11, s5
	s_cselect_b32 s3, s12, s3
	s_add_i32 s11, s5, 1
	s_cmp_ge_u32 s3, s6
	s_cselect_b32 s3, s11, s5
	s_xor_b32 s23, s3, s22
	s_sub_i32 s6, s23, s22
	s_mul_i32 s3, s6, s4
	s_sub_i32 s2, s2, s3
	s_add_i32 s2, s2, s7
	s_lshl_b32 s11, s2, 5
	v_lshlrev_b32_e32 v1, 4, v0
	v_or_b32_e32 v18, s11, v17
	v_and_b32_e32 v22, 0x70, v1
	v_cmp_gt_i32_e64 s[4:5], s8, v18
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_mad_u64_u32 v[4:5], s[12:13], v18, s10, v[22:23]
	v_ashrrev_i32_e32 v5, 31, v4
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[4:5], s[16:17], 0, v[4:5]
	global_load_dwordx4 v[4:7], v[4:5], off
.LBB0_2:
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx2 s[18:19], s[0:1], 0x8
	s_lshl_b32 s12, s6, 5
	s_movk_i32 s2, 0x90
	v_mad_u32_u24 v1, v17, s2, v22
	v_or_b32_e32 v19, s12, v17
	s_waitcnt vmcnt(0)
	ds_write_b128 v1, v[4:7]
	v_cmp_gt_i32_e64 s[2:3], s9, v19
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB0_4
; %bb.3:
	v_mad_u64_u32 v[2:3], s[14:15], v19, s10, v[22:23]
	v_ashrrev_i32_e32 v3, 31, v2
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], s[18:19], 0, v[2:3]
	global_load_dwordx4 v[2:5], v[2:3], off
.LBB0_4:
	s_or_b64 exec, exec, s[6:7]
	v_and_b32_e32 v20, 31, v0
	s_cmpk_gt_i32 s10, 0x7f
	v_or_b32_e32 v23, s12, v20
	s_waitcnt vmcnt(0)
	ds_write_b128 v1, v[2:5] offset:9216
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_6
; %bb.5:                                ; %..preheader243_crit_edge
	v_or_b32_e32 v16, s12, v20
	s_mov_b64 s[12:13], 0
	s_branch .LBB0_7
.LBB0_6:
	s_mov_b64 s[12:13], -1
                                        ; implicit-def: $vgpr16
.LBB0_7:                                ; %Flow447
	s_load_dwordx2 s[6:7], s[0:1], 0x20
	v_bfe_u32 v24, v0, 5, 1
	s_mov_b32 s20, 1
	s_andn2_b64 vcc, exec, s[12:13]
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	s_cbranch_vccnz .LBB0_17
; %bb.8:                                ; %.lr.ph
	s_load_dwordx4 s[12:15], s[0:1], 0x10
	s_ashr_i32 s0, s10, 31
	s_lshr_b32 s1, s0, 25
	s_add_i32 s1, s10, s1
	s_ashr_i32 s21, s1, 7
	s_lshr_b32 s0, s0, 28
	v_mul_lo_u32 v1, v18, s10
	v_mul_lo_u32 v3, v19, s10
	s_movk_i32 s1, 0x80
	s_add_i32 s0, s10, s0
	v_add3_u32 v26, v3, v22, s1
	v_add3_u32 v28, v1, v22, s1
	v_lshl_or_b32 v1, s23, 5, v20
	s_lshl_b32 s1, s22, 5
	s_ashr_i32 s0, s0, 4
	v_subrev_u32_e32 v1, s1, v1
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v2, 2, v24
	v_or_b32_e32 v4, 4, v24
	v_or_b32_e32 v6, 6, v24
	v_mul_lo_u32 v30, s0, v1
	v_add_u32_e32 v1, s11, v20
	v_lshlrev_b32_e32 v25, 4, v24
	v_mul_lo_u32 v32, s0, v1
	s_mov_b32 s10, 0
	v_mul_u32_u24_e32 v34, 0x90, v17
	v_mul_u32_u24_e32 v35, 0x90, v20
	v_lshlrev_b32_e32 v36, 3, v24
	s_movk_i32 s22, 0x101
	v_lshlrev_b32_e32 v37, 3, v2
	v_lshlrev_b32_e32 v38, 3, v4
	v_lshlrev_b32_e32 v39, 3, v6
	s_mov_b32 s23, 0
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	s_branch .LBB0_11
.LBB0_9:                                ;   in Loop: Header=BB0_11 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_addk_i32 s24, 0x2400
	v_add3_u32 v20, s24, v34, v22
	s_waitcnt vmcnt(0)
	ds_write_b128 v20, v[16:19]
.LBB0_10:                               ;   in Loop: Header=BB0_11 Depth=1
	v_ashrrev_i32_e32 v33, 31, v32
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[16:17], s[12:13], 0, v[32:33]
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u64 v[18:19], s[14:15], 0, v[30:31]
	global_load_dwordx2 v[20:21], v[16:17], off
	global_load_dwordx2 v[44:45], v[18:19], off
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_mul_i32 s0, s10, 0x1200
	v_add3_u32 v27, s0, v35, v25
	s_addk_i32 s0, 0x2400
	v_add3_u32 v29, s0, v35, v25
	ds_read_b128 v[16:19], v27
	ds_read_b128 v[40:43], v29
	s_waitcnt vmcnt(1)
	v_bfe_u32 v31, v20, v36, 8
	s_waitcnt vmcnt(0)
	v_bfe_u32 v33, v44, v36, 8
	v_mul_u32_u24_e32 v31, 0x101, v31
	v_mul_u32_u24_e32 v33, 0x101, v33
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[40:43], v[0:15], v31, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[16:19], v27 offset:32
	ds_read_b128 v[40:43], v29 offset:32
	v_lshrrev_b64 v[46:47], v37, v[20:21]
	v_lshrrev_b64 v[48:49], v37, v[44:45]
	v_mul_u32_u24_sdwa v31, v46, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_mul_u32_u24_sdwa v33, v48, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[40:43], v[0:15], v31, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[16:19], v27 offset:64
	ds_read_b128 v[40:43], v29 offset:64
	v_lshrrev_b64 v[46:47], v38, v[20:21]
	v_lshrrev_b64 v[48:49], v38, v[44:45]
	v_mul_u32_u24_sdwa v31, v46, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_mul_u32_u24_sdwa v33, v48, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[40:43], v[0:15], v31, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[16:19], v27 offset:96
	ds_read_b128 v[40:43], v29 offset:96
	v_lshrrev_b64 v[20:21], v39, v[20:21]
	v_lshrrev_b64 v[44:45], v39, v[44:45]
	v_mul_u32_u24_sdwa v20, v20, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	v_mul_u32_u24_sdwa v21, v44, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[40:43], v[0:15], v20, v21 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	s_setprio 0
	s_xor_b32 s10, s10, 1
	s_xor_b32 s20, s20, 1
	v_add_u32_e32 v26, 0x80, v26
	v_add_u32_e32 v28, 0x80, v28
	v_add_u32_e32 v30, 8, v30
	s_cmp_eq_u32 s21, s23
	v_add_u32_e32 v32, 8, v32
	s_barrier
	s_cbranch_scc1 .LBB0_16
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	s_add_i32 s23, s23, 1
	s_cmp_ge_i32 s23, s21
	s_cbranch_scc1 .LBB0_10
; %bb.12:                               ;   in Loop: Header=BB0_11 Depth=1
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_11 Depth=1
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshl_add_u64 v[18:19], s[16:17], 0, v[28:29]
	global_load_dwordx4 v[18:21], v[18:19], off
.LBB0_14:                               ;   in Loop: Header=BB0_11 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_mul_i32 s24, s20, 0x1200
	v_add3_u32 v17, s24, v34, v22
	s_waitcnt vmcnt(0)
	ds_write_b128 v17, v[18:21]
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_9
; %bb.15:                               ;   in Loop: Header=BB0_11 Depth=1
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[16:17], s[18:19], 0, v[26:27]
	global_load_dwordx4 v[16:19], v[16:17], off
	s_branch .LBB0_9
.LBB0_16:                               ; %.preheader243.loopexit
	v_mov_b32_e32 v16, v23
.LBB0_17:                               ; %Flow448
	v_lshl_or_b32 v17, v24, 2, s11
	v_cmp_gt_i32_e32 vcc, s9, v16
	v_cmp_gt_i32_e64 s[0:1], s8, v17
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_19
; %bb.18:
	v_bfe_u32 v18, v0, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v0, v18, s2
	v_mad_u64_u32 v[18:19], s[2:3], v17, s9, v[16:17]
	v_ashrrev_i32_e32 v19, 31, v18
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	global_store_short_d16_hi v[18:19], v0, off
.LBB0_19:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 1, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_21
; %bb.20:
	v_bfe_u32 v18, v1, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v18, v1, v18, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v18, off
.LBB0_21:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 2, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_23
; %bb.22:
	v_bfe_u32 v1, v2, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v2, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_23:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 3, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_bfe_u32 v1, v3, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v3, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_25:                               ; %.preheader.1
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 8, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_bfe_u32 v1, v4, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v4, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_27:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 9, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_bfe_u32 v1, v5, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v5, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_29:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 10, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_bfe_u32 v1, v6, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v6, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_31:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 11, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_bfe_u32 v1, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v7, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_33:                               ; %.preheader.2
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 16, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_bfe_u32 v1, v8, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v8, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_35:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 17, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_bfe_u32 v1, v9, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v9, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_37:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 18, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_39
; %bb.38:
	v_bfe_u32 v1, v10, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v10, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_39:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 19, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_41
; %bb.40:
	v_bfe_u32 v1, v11, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v11, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_41:                               ; %.preheader.3
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 24, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_43
; %bb.42:
	v_bfe_u32 v1, v12, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v12, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_43:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 25, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_45
; %bb.44:
	v_bfe_u32 v1, v13, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v13, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_45:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 26, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[2:3], s[0:1], vcc
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_47
; %bb.46:
	v_bfe_u32 v1, v14, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v14, v1, s2
	v_mad_u64_u32 v[0:1], s[2:3], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_47:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 27, v17
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_and_b64 s[0:1], s[0:1], vcc
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_bfe_u32 v1, v15, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v2, v15, v1, s0
	v_mad_u64_u32 v[0:1], s[0:1], v0, s9, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	global_store_short_d16_hi v[0:1], v2, off
.LBB0_49:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fp4_lds_opt
		.amdhsa_group_segment_fixed_size 18432
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 312
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
		.amdhsa_next_free_vgpr 50
		.amdhsa_next_free_sgpr 25
		.amdhsa_accum_offset 52
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
	.size	fp4_lds_opt, .Lfunc_end0-fp4_lds_opt
                                        ; -- End function
	.set fp4_lds_opt.num_vgpr, 50
	.set fp4_lds_opt.num_agpr, 0
	.set fp4_lds_opt.numbered_sgpr, 25
	.set fp4_lds_opt.private_seg_size, 0
	.set fp4_lds_opt.uses_vcc, 1
	.set fp4_lds_opt.uses_flat_scratch, 0
	.set fp4_lds_opt.has_dyn_sized_stack, 0
	.set fp4_lds_opt.has_recursion, 0
	.set fp4_lds_opt.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2836
; TotalNumSgprs: 31
; NumVgprs: 50
; NumAgprs: 0
; TotalNumVgprs: 50
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 18432 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 31
; NumVGPRsForWavesPerEU: 50
; AccumOffset: 52
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 12
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_33e1a9c3232aa6e5,@object ; @__hip_cuid_33e1a9c3232aa6e5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_33e1a9c3232aa6e5
__hip_cuid_33e1a9c3232aa6e5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_33e1a9c3232aa6e5, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_33e1a9c3232aa6e5
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
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         60
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         68
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         70
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         72
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         74
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         76
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         78
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         120
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 18432
    .kernarg_segment_align: 8
    .kernarg_segment_size: 312
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           fp4_lds_opt
    .private_segment_fixed_size: 0
    .sgpr_count:     31
    .sgpr_spill_count: 0
    .symbol:         fp4_lds_opt.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     50
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
