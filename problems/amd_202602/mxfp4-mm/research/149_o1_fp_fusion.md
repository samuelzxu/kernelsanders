# #149 O1 + fp_fusion=True
Results slightly worse than #148 O1 alone.
fp_fusion=True changes codegen path that negates O1's register pressure benefit.
Keep #148 (O1 only, no fp_fusion change).
