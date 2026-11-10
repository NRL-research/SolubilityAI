"""Step 07: ONE-TIME held-out test evaluation of every saved configuration.

1. Unified CV criterion: every model's out-of-fold RMSE on the IN-DOMAIN training molecules
   (data/06b_train_v2.csv; identical folds for v1- and v2-trained models) -> comparable across
   training-set versions and blends.
2. Pre-registration (written BEFORE the test file is read): selection rule = lowest in-domain
   CV RMSE; frozen in results/07_preregistration.json together with the list of all evaluated
   configurations and the primary/secondary test sets.
3. Each saved model is loaded from disk through its own predict.py (the exact UI path) and
   predicts all test SMILES once.  Primary test = in-domain test (data/06b_test_v2.csv, n=1804);
   secondary = full curated test (data/06_test.csv, n=1852, includes metal/mixture entries).
4. 95% bootstrap CIs (molecule-level and cluster-group-level; 2000 resamples; seeds 7/8);
   paired bootstrap of the pre-registered model vs every other model (same resamples).
5. Subgroups for the pre-registered model; external SC sets (caveated).
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

res = pd.read_csv(R / "results/cv_results.csv")
res = res[res.name.map(lambda n: (R / f"saved_models/{n}/config.json").exists())].copy()
dom_tr = pd.read_csv(R / "data/06b_train_v2.csv")[["mol_id"]]
def cv_indomain(n):
    o = dom_tr.merge(pd.read_csv(R / f"results/oof/{n}.csv"), on="mol_id", how="left")
    assert o.oof.notna().all(), n
    return runner.metrics(o.y, o.oof)
cvd = {n: cv_indomain(n) for n in res.name}
res["cv_rmse_indomain"] = res.name.map(lambda n: cvd[n]["rmse"]); res["cv_r2_indomain"] = res.name.map(lambda n: cvd[n]["r2"])
res = res.sort_values("cv_rmse_indomain")
res.to_csv(R / "results/07_cv_indomain.csv", index=False)

prereg = R / "results/07_preregistration.json"
if prereg.exists():
    raise SystemExit("Test set already evaluated (preregistration exists) -- refusing to re-run. "
                     "Delete results/07_* deliberately if a re-run is intended, and report it.")
BEST = res.name.iloc[0]
json.dump(dict(selection_rule="lowest pooled out-of-fold RMSE on in-domain training molecules (5-fold grouped CV)",
               selected=BEST, selected_cv_rmse_indomain=float(res.cv_rmse_indomain.iloc[0]),
               primary_test="data/06b_test_v2.csv (in-domain)", secondary_test="data/06_test.csv (full curated)",
               evaluated_configs=res.name.tolist()), open(prereg, "w"), indent=2)
print("pre-registered best:", BEST, flush=True)

def load(folder):
    spec = importlib.util.spec_from_file_location(f"pred_{folder.name}", folder / "predict.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.load_predictor(folder)

te = pd.read_csv(R / "data/06_test.csv")
dom_te = set(pd.read_csv(R / "data/06b_test_v2.csv").mol_id)
te["in_domain"] = te.mol_id.isin(dom_te)
preds = {}
for n in res.name:
    preds[n] = load(R / "saved_models" / n)(te.SMILES.tolist())
    assert np.isfinite(preds[n]).all(), n
pd.DataFrame(dict(mol_id=te.mol_id, group=te.group, in_domain=te.in_domain, y=te.LogS, **preds)).to_csv(
    R / "results/07_test_predictions.csv", index=False)

def rmse(y, p): return np.sqrt(np.mean((y - p) ** 2))
def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
B = 2000
all_rows = []
for tset, mask in [("indomain", te.in_domain.values), ("full", np.ones(len(te), bool))]:
    y = te.LogS.values[mask]; groups = te.group.values[mask]
    rng = np.random.RandomState(7); bi = [rng.randint(0, len(y), len(y)) for _ in range(B)]
    ug = np.unique(groups); gidx = {g: np.where(groups == g)[0] for g in ug}
    rng2 = np.random.RandomState(8); ci = [np.concatenate([gidx[g] for g in rng2.choice(ug, len(ug))]) for _ in range(B)]
    pb = preds[BEST][mask]
    for n in res.name:
        p = preds[n][mask]; m = runner.metrics(y, p)
        br = np.array([rmse(y[i], p[i]) for i in bi]); b2 = np.array([r2(y[i], p[i]) for i in bi])
        cr = np.array([rmse(y[i], p[i]) for i in ci]); c2 = np.array([r2(y[i], p[i]) for i in ci])
        d = np.array([rmse(y[i], p[i]) - rmse(y[i], pb[i]) for i in bi])
        dc = np.array([rmse(y[i], p[i]) - rmse(y[i], pb[i]) for i in ci])
        all_rows.append(dict(test_set=tset, n_test=int(mask.sum()), name=n,
            cv_rmse_indomain=float(res.set_index("name").cv_rmse_indomain[n]), cv_r2_indomain=float(res.set_index("name").cv_r2_indomain[n]),
            test_rmse=m["rmse"], test_r2=m["r2"], test_mae=m["mae"],
            rmse_ci_mol=[float(x) for x in np.percentile(br, [2.5, 97.5])], r2_ci_mol=[float(x) for x in np.percentile(b2, [2.5, 97.5])],
            rmse_ci_cluster=[float(x) for x in np.percentile(cr, [2.5, 97.5])], r2_ci_cluster=[float(x) for x in np.percentile(c2, [2.5, 97.5])],
            d_rmse_vs_best=float(m["rmse"] - rmse(y, pb)),
            d_rmse_ci_mol=[float(x) for x in np.percentile(d, [2.5, 97.5])],
            p_other_at_least_as_good_mol=None if n == BEST else float(np.mean(d <= 0)),
            d_rmse_ci_cluster=[float(x) for x in np.percentile(dc, [2.5, 97.5])],
            p_other_at_least_as_good_cluster=None if n == BEST else float(np.mean(dc <= 0))))
tm = pd.DataFrame(all_rows)
tm.to_json(R / "results/07_test_metrics.json", orient="records", indent=1)
tm[tm.test_set == "indomain"].drop(columns="test_set").to_csv(R / "results/07_test_metrics.csv", index=False)
tm[tm.test_set == "full"].drop(columns="test_set").to_csv(R / "results/07_test_metrics_fulltest.csv", index=False)

# ---------------- subgroups for the pre-registered model (in-domain test) ----------------
tr = pd.read_csv(R / "data/06_train.csv")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
tfp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.parent_smiles]
nn = np.array([max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), tfp)) for s in te.parent_smiles])
p, y, dm = preds[BEST], te.LogS.values, te.in_domain.values
masks = {"all_in_domain": dm, "out_of_domain(metal/mixture)": ~dm,
         "acyclic": dm & ~te.has_ring_parent.values, "ring": dm & te.has_ring_parent.values,
         "salt/multicomponent": dm & te.is_multicomponent.values, "single_component": dm & ~te.is_multicomponent.values,
         "nn_sim<0.4": dm & (nn < 0.4), "0.4<=nn_sim<0.6": dm & (nn >= 0.4) & (nn < 0.6), "nn_sim>=0.6": dm & (nn >= 0.6),
         "logS<-6": dm & (y < -6), "-6<=logS<-2": dm & (y >= -6) & (y < -2), "logS>=-2": dm & (y >= -2)}
sub = {k: (dict(n=int(mk.sum()), **runner.metrics(y[mk], p[mk])) if mk.sum() > 2 else dict(n=int(mk.sum()))) for k, mk in masks.items()}
json.dump(dict(model=BEST, subgroups=sub), open(R / "results/07_subgroups_best.json", "w"), indent=2)

# ---------------- external SC sets (caveated: logS0 intrinsic solubility; SC1 row alignment suspect) ----------------
ext = {}
for f, col in [("SC2_tight_processed", "logS0"), ("SC2_loose_processed", "logS0"), ("SC1_processed", "logS0_uM")]:
    s = pd.read_csv(R / f"data/00_raw/{f}.csv")
    s = s[pd.to_numeric(s[col], errors="coerce").notna()].copy(); s[col] = s[col].astype(float)
    ext[f] = {"n": int(len(s))}
    for n in res.name:
        pp = load(R / "saved_models" / n)(s.canonical_smiles.tolist())
        ext[f][n] = runner.metrics(s[col].values, pp)
json.dump(ext, open(R / "results/07_external_sc.json", "w"), indent=2)
show = tm[tm.test_set == "indomain"][["name", "cv_rmse_indomain", "test_rmse", "test_r2", "d_rmse_vs_best", "p_other_at_least_as_good_mol", "p_other_at_least_as_good_cluster"]]
print(show.to_string()); print(json.dumps(sub, indent=1))
