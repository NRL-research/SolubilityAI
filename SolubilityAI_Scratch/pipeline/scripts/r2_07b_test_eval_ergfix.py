"""Round 2, step 07b (bug fix): ONE-TIME test evaluation of the three deterministic-ErG replacements.

Why: verification (scripts/r2_08_verify.py) showed that r2_lgbm_d_av_erg and
r2_lgbm_desc_mordred_avalon_erg do not reproduce their recorded test metrics exactly, because
RDKit's ErG fingerprint depends on state cached on the Mol object (fixed by featurize.erg2).  The
pre-registered round-2 verdict (results/r2_07_verdict.json) is NOT changed by this step.
Pre-registration (written before the test file is read): the bug-fixed candidate
r2_blend_nnls_ergfix is compared with blend_nnls using the same one-sided paired cluster
bootstrap (2000 resamples, seed 8, alpha 0.05) as round 2.
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner

NEW = ["r2_lgbm_d_av_erg2", "r2_lgbm_desc_mordred_avalon_erg2", "r2_blend_nnls_ergfix"]
pre = R / "results/r2_07b_preregistration.json"
if pre.exists(): raise SystemExit("already evaluated")
dom_tr = pd.read_csv(R / "data/06b_train_v2.csv")[["mol_id"]]
C = {n: runner.metrics(*(lambda o: (o.y, o.oof))(dom_tr.merge(pd.read_csv(R / f"results/oof/{n}.csv"), on="mol_id"))) for n in NEW}
json.dump(dict(purpose="bug-fix replacements for the ErG models (non-deterministic featurization)", configs=NEW,
               cv_rmse_indomain={n: C[n]["rmse"] for n in NEW}, candidate="r2_blend_nnls_ergfix", reference="blend_nnls",
               primary_test="one-sided paired cluster bootstrap on in-domain test RMSE, 2000 resamples, seed 8, alpha 0.05",
               note="does not replace the pre-registered round-2 verdict"), open(pre, "w"), indent=2)
te = pd.read_csv(R / "data/06_test.csv")
old = pd.read_csv(R / "results/07_test_predictions.csv"); assert (old.mol_id.values == te.mol_id.values).all()
dom_te = set(pd.read_csv(R / "data/06b_test_v2.csv").mol_id); te["in_domain"] = te.mol_id.isin(dom_te)
def load(folder):
    spec = importlib.util.spec_from_file_location(f"pred_{folder.name}", folder / "predict.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.load_predictor(folder)
preds = {"blend_nnls": old["blend_nnls"].values}
for n in NEW:
    preds[n] = load(R / "saved_models" / n)(te.SMILES.tolist()); assert np.isfinite(preds[n]).all()
pd.DataFrame(dict(mol_id=te.mol_id, in_domain=te.in_domain, y=te.LogS, **{n: preds[n] for n in NEW})).to_csv(R / "results/r2_07b_test_predictions.csv", index=False)
def rmse(y, p): return np.sqrt(np.mean((y - p) ** 2))
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
rows = []
for tset, mask in [("indomain", te.in_domain.values), ("full", np.ones(len(te), bool))]:
    y = te.LogS.values[mask]; g = te.group.values[mask]
    rng = np.random.RandomState(7); bi = [rng.randint(0, len(y), len(y)) for _ in range(2000)]
    ug = np.unique(g); gi = {k: np.where(g == k)[0] for k in ug}
    rng2 = np.random.RandomState(8); ci = [np.concatenate([gi[k] for k in rng2.choice(ug, len(ug))]) for _ in range(2000)]
    ref = preds["blend_nnls"][mask]
    for n in NEW:
        p = preds[n][mask]; m = runner.metrics(y, p)
        d = np.array([rmse(y[i], p[i]) - rmse(y[i], ref[i]) for i in bi]); dc = np.array([rmse(y[i], p[i]) - rmse(y[i], ref[i]) for i in ci])
        rows.append(dict(test_set=tset, n_test=int(mask.sum()), name=n, round=2, cv_rmse_indomain=C[n]["rmse"], cv_r2_indomain=C[n]["r2"],
            test_rmse=m["rmse"], test_r2=m["r2"], test_mae=m["mae"],
            rmse_ci_mol=[float(x) for x in np.percentile([rmse(y[i], p[i]) for i in bi], [2.5, 97.5])],
            rmse_ci_cluster=[float(x) for x in np.percentile([rmse(y[i], p[i]) for i in ci], [2.5, 97.5])],
            r2_ci_mol=[float(x) for x in np.percentile([r2(y[i], p[i]) for i in bi], [2.5, 97.5])],
            r2_ci_cluster=[float(x) for x in np.percentile([r2(y[i], p[i]) for i in ci], [2.5, 97.5])],
            d_rmse_vs_blend_nnls=float(m["rmse"] - rmse(y, ref)), d_ci_mol=[float(x) for x in np.percentile(d, [2.5, 97.5])],
            d_ci_cluster=[float(x) for x in np.percentile(dc, [2.5, 97.5])],
            p_not_better_than_blend_nnls_mol=float(np.mean(d >= 0)), p_not_better_than_blend_nnls_cluster=float(np.mean(dc >= 0))))
tm = pd.DataFrame(rows); tm.to_json(R / "results/r2_07b_test_metrics.json", orient="records", indent=1)
print(tm[["test_set", "name", "cv_rmse_indomain", "test_rmse", "test_r2", "d_rmse_vs_blend_nnls", "d_ci_cluster", "p_not_better_than_blend_nnls_cluster", "p_not_better_than_blend_nnls_mol"]].to_string())
