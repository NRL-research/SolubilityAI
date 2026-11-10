"""Round 2 exploratory (CV-only, nested): 'gated' NNLS blends whose weights differ by molecule
type - separate NNLS weights per nearest-neighbour-similarity band, or per size band (MolWt).
Motivation: error analysis showed errors depend strongly on novelty and size, so the best mix of
models might too.  Not saved unless it beats the single NNLS blend in nested CV."""
import sys; sys.argv = ["x"]
src = open(__file__.replace("r2_06c_gated_blend_test.py", "r2_06_stack.py")).read().split("all_rows = np.arange")[0]
exec(src)
def gated(edges, col):
    oof = np.zeros(len(y))
    for k in range(5):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        bt, bv = np.digitize(META[tr, col], edges), np.digitize(META[va, col], edges)
        wg, bg = fit_nnls(P[tr], y[tr])
        for g in np.unique(bv):
            rows = tr[bt == g]
            w, b = fit_nnls(P[rows], y[rows]) if len(rows) > 300 else (wg, bg)
            oof[va[bv == g]] = P[va[bv == g]] @ w + b
    return runner.metrics(y, oof)
print("gate by nn_sim [0.4,0.6]:", gated([0.4, 0.6], 6))
print("gate by MolWt [200,350]:", gated([200, 350], 0))
# Recorded result (2026-10-02): nn_sim-gated 0.8862, MolWt-gated 0.8888 vs single r2_blend_nnls 0.8829 -> worse; not saved.
