"""Round 2, step 07: ONE-TIME held-out test evaluation of the round-2 configurations.

Protocol (identical in spirit to round 1, scripts/07_test_eval.py):
1. Unified CV criterion = OOF RMSE on the 7,108 in-domain training molecules (same folds).
2. Pre-registration, written BEFORE the test file is read (results/r2_07_preregistration.json):
   * candidate = configuration with the lowest in-domain CV RMSE among ALL round-1 + round-2 configs;
   * primary hypothesis: candidate has lower in-domain test RMSE than the round-1 final model
     blend_nnls, tested with a one-sided paired CLUSTER bootstrap (2000 resamples, seed 8), alpha 0.05;
   * the candidate replaces blend_nnls as "current best" only if that test is passed.
3. Only round-2 configurations are predicted here (each once, through its saved predict.py).
   Round-1 test predictions are re-used from results/07_test_predictions.csv, never recomputed.
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner

res = pd.read_csv(R / "results/cv_results.csv")
res = res[res.name.map(lambda n: (R / f"saved_models/{n}/config.json").exists())].copy()
r1 = set(pd.read_csv(R / "results/07_cv_indomain.csv").name)
dom_tr = pd.read_csv(R / "data/06b_train_v2.csv")[["mol_id"]]
def cvd(n):
    o = dom_tr.merge(pd.read_csv(R / f"results/oof/{n}.csv"), on="mol_id", how="left"); assert o.oof.notna().all(), n
    return runner.metrics(o.y, o.oof)
C = {n: cvd(n) for n in res.name}
res["cv_rmse_indomain"] = res.name.map(lambda n: C[n]["rmse"]); res["cv_r2_indomain"] = res.name.map(lambda n: C[n]["r2"])
res["cv_mae_indomain"] = res.name.map(lambda n: C[n]["mae"]); res["round"] = np.where(res.name.isin(r1), 1, 2)
res = res.sort_values("cv_rmse_indomain"); res.to_csv(R / "results/r2_07_cv_indomain_all.csv", index=False)
new = res[res["round"] == 2].name.tolist()

prereg = R / "results/r2_07_preregistration.json"
if prereg.exists():
    raise SystemExit("Round-2 test evaluation already done (preregistration exists); refusing to re-run.")
CAND = res.name.iloc[0]
json.dump(dict(selection_rule="lowest in-domain 5-fold grouped CV RMSE among all round-1 and round-2 configurations",
               candidate=CAND, candidate_cv_rmse_indomain=float(res.cv_rmse_indomain.iloc[0]),
               reference="blend_nnls (round-1 final)", reference_cv_rmse_indomain=float(res.set_index("name").cv_rmse_indomain["blend_nnls"]),
               primary_test="one-sided paired cluster bootstrap on in-domain test RMSE, 2000 resamples, seed 8, alpha 0.05",
               round2_configs_evaluated=new), open(prereg, "w"), indent=2)
print("pre-registered candidate:", CAND, flush=True)

def load(folder):
    spec = importlib.util.spec_from_file_location(f"pred_{folder.name}", folder / "predict.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.load_predictor(folder)

te = pd.read_csv(R / "data/06_test.csv")
old = pd.read_csv(R / "results/07_test_predictions.csv")
assert (old.mol_id.values == te.mol_id.values).all()
dom_te = set(pd.read_csv(R / "data/06b_test_v2.csv").mol_id); te["in_domain"] = te.mol_id.isin(dom_te)
preds = {n: old[n].values for n in old.columns if n in r1}
for n in new:
    preds[n] = load(R / "saved_models" / n)(te.SMILES.tolist()); assert np.isfinite(preds[n]).all(), n
    print(n, runner.metrics(te.LogS.values[te.in_domain.values], preds[n][te.in_domain.values]), flush=True)
pd.DataFrame(dict(mol_id=te.mol_id, group=te.group, in_domain=te.in_domain, y=te.LogS, **{n: preds[n] for n in new})).to_csv(
    R / "results/r2_07_test_predictions.csv", index=False)

def rmse(y, p): return np.sqrt(np.mean((y - p) ** 2))
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
B = 2000; rows = []
for tset, mask in [("indomain", te.in_domain.values), ("full", np.ones(len(te), bool))]:
    y = te.LogS.values[mask]; g = te.group.values[mask]
    rng = np.random.RandomState(7); bi = [rng.randint(0, len(y), len(y)) for _ in range(B)]
    ug = np.unique(g); gi = {k: np.where(g == k)[0] for k in ug}
    rng2 = np.random.RandomState(8); ci = [np.concatenate([gi[k] for k in rng2.choice(ug, len(ug))]) for _ in range(B)]
    ref = preds["blend_nnls"][mask]
    for n in new + ["blend_nnls"]:
        p = preds[n][mask]; m = runner.metrics(y, p)
        d = np.array([rmse(y[i], p[i]) - rmse(y[i], ref[i]) for i in bi]); dc = np.array([rmse(y[i], p[i]) - rmse(y[i], ref[i]) for i in ci])
        rows.append(dict(test_set=tset, n_test=int(mask.sum()), name=n, round=1 if n in r1 else 2,
            cv_rmse_indomain=C[n]["rmse"], cv_r2_indomain=C[n]["r2"], test_rmse=m["rmse"], test_r2=m["r2"], test_mae=m["mae"],
            rmse_ci_mol=[float(x) for x in np.percentile([rmse(y[i], p[i]) for i in bi], [2.5, 97.5])],
            rmse_ci_cluster=[float(x) for x in np.percentile([rmse(y[i], p[i]) for i in ci], [2.5, 97.5])],
            r2_ci_mol=[float(x) for x in np.percentile([r2(y[i], p[i]) for i in bi], [2.5, 97.5])],
            r2_ci_cluster=[float(x) for x in np.percentile([r2(y[i], p[i]) for i in ci], [2.5, 97.5])],
            d_rmse_vs_blend_nnls=float(m["rmse"] - rmse(y, ref)),
            d_ci_mol=[float(x) for x in np.percentile(d, [2.5, 97.5])], d_ci_cluster=[float(x) for x in np.percentile(dc, [2.5, 97.5])],
            p_not_better_than_blend_nnls_mol=float(np.mean(d >= 0)), p_not_better_than_blend_nnls_cluster=float(np.mean(dc >= 0))))
tm = pd.DataFrame(rows)
tm.to_json(R / "results/r2_07_test_metrics.json", orient="records", indent=1)
tm[tm.test_set == "indomain"].drop(columns="test_set").to_csv(R / "results/r2_07_test_metrics.csv", index=False)
tm[tm.test_set == "full"].drop(columns="test_set").to_csv(R / "results/r2_07_test_metrics_fulltest.csv", index=False)
c = tm[(tm.test_set == "indomain") & (tm.name == CAND)].iloc[0] if CAND in new else None
verdict = dict(candidate=CAND)
if c is not None:
    verdict.update(candidate_test_rmse=c.test_rmse, candidate_test_r2=c.test_r2, d_rmse=c.d_rmse_vs_blend_nnls, d_ci_cluster=c.d_ci_cluster,
                   d_ci_mol=c.d_ci_mol, p_cluster=c.p_not_better_than_blend_nnls_cluster, p_mol=c.p_not_better_than_blend_nnls_mol,
                   passed=bool(c.p_not_better_than_blend_nnls_cluster < 0.05))
else:
    verdict.update(passed=False, note="candidate is a round-1 model; nothing new to test")
json.dump(verdict, open(R / "results/r2_07_verdict.json", "w"), indent=2, default=float)
print(tm[tm.test_set == "indomain"][["name", "cv_rmse_indomain", "test_rmse", "test_r2", "d_rmse_vs_blend_nnls", "p_not_better_than_blend_nnls_cluster"]].sort_values("cv_rmse_indomain").to_string())
print(json.dumps(verdict, indent=1, default=float))

# subgroup breakdown of the candidate (in-domain test), as in round 1
tr = pd.read_csv(R / "data/06_train.csv")
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
tfp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.parent_smiles]
nn = np.array([max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), tfp)) for s in te.parent_smiles])
p, y, dm = preds[CAND], te.LogS.values, te.in_domain.values
masks = {"all_in_domain": dm, "out_of_domain": ~dm, "acyclic": dm & ~te.has_ring_parent.values, "ring": dm & te.has_ring_parent.values,
         "nn_sim<0.4": dm & (nn < 0.4), "0.4<=nn_sim<0.6": dm & (nn >= 0.4) & (nn < 0.6), "nn_sim>=0.6": dm & (nn >= 0.6),
         "logS<-6": dm & (y < -6), "-6<=logS<-2": dm & (y >= -6) & (y < -2), "logS>=-2": dm & (y >= -2)}
json.dump(dict(model=CAND, subgroups={k: dict(n=int(mk.sum()), **runner.metrics(y[mk], p[mk])) for k, mk in masks.items()}),
          open(R / "results/r2_07_subgroups_candidate.json", "w"), indent=2)
ext = {}
for f, col in [("SC2_tight_processed", "logS0"), ("SC2_loose_processed", "logS0")]:
    s = pd.read_csv(R / f"data/00_raw/{f}.csv")
    ext[f] = dict(n=len(s), **{n: runner.metrics(s[col].values, load(R / "saved_models" / n)(s.canonical_smiles.tolist())) for n in [CAND]})
json.dump(ext, open(R / "results/r2_07_external_sc.json", "w"), indent=2)
