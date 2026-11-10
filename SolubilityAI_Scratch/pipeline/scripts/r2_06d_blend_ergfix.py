"""Round 2 bug fix: rebuild the round-2 NNLS blend with the deterministic-ErG models.
Identical to r2_blend_nnls in scripts/r2_06_stack.py (nested CV, NNLS over the full pool) except that
the two models using the non-deterministic 'erg' block are excluded from the pool and their
'erg2' retrains (r2_lgbm_d_av_erg2, r2_lgbm_desc_mordred_avalon_erg2) are included."""
import sys
EXCL = "r2_lgbm_d_av_erg,r2_lgbm_desc_mordred_avalon_erg,r2_blend_nnls_ergfix"
sys.argv = ["r2_06_stack.py", EXCL]
__file__ = __file__.replace("r2_06d_blend_ergfix.py", "r2_06_stack.py")
src = open(__file__).read().split("all_rows = np.arange")[0]
exec(src)
assert "r2_lgbm_d_av_erg2" in single and "r2_lgbm_desc_mordred_avalon_erg2" in single
assert not any(n in single for n in EXCL.split(","))
all_rows = np.arange(len(y))
oof, per = nested(NNLSBlend)
save("r2_blend_nnls_ergfix", "nnls", NNLSBlend().fit(all_rows), oof, per,
     "Bug-fix rebuild of r2_blend_nnls: same method, ErG models replaced by deterministic erg2 retrains.")
