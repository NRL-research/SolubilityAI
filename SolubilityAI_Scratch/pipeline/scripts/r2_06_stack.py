"""Round 2, step 06: stacking (learned meta-models) and an updated NNLS blend.

All fitted on out-of-fold (OOF) predictions of the IN-DOMAIN training molecules, scored with
NESTED CV (for outer fold k, everything - pool choice, weights, meta-model - is fitted on the
other 4 folds' OOF rows only).  The test set is not read.

Stackers
  r2_blend_nnls        NNLS over the full round-1 + round-2 pool (round-1 method, bigger pool)
  r2_stack_ridge       ridge regression (alpha by inner 4-fold CV) over the family-best pool
  r2_stack_lgbm_meta   small LightGBM over [family-best pool predictions + 6 RDKit descriptors +
                       nn_sim]; lets the combination depend on molecule type / novelty
nn_sim for an OOF row = max Tanimoto to in-domain training molecules in OTHER folds (what the
fold models saw); for new molecules = max Tanimoto to all in-domain training molecules.
Known caveat (shared with round-1 blends): base-model OOF rows used to fit the stacker came from
fold models that saw the outer fold, a mild optimistic bias; the one-time test is the real check.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

EXCLUDE = set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else set()
res = pd.read_csv(R / "results/cv_results.csv")
single = [n for n in res.name if (R / f"results/oof/{n}.csv").exists() and not n.startswith(("blend", "r2_blend", "r2_stack"))
          and (R / f"saved_models/{n}/config.json").exists() and n not in EXCLUDE]
dom = pd.read_csv(R / "data/06b_train_v2.csv")
O = {n: dom[["mol_id"]].merge(pd.read_csv(R / f"results/oof/{n}.csv"), on="mol_id", how="left") for n in single}
for n in single:
    assert O[n].oof.notna().all() and (O[n].cv_fold.values == dom.cv_fold.values).all(), n
y, folds = dom.LogS.values, dom.cv_fold.values
assert np.allclose(O[single[0]].y.values, y)
P = np.column_stack([O[n].oof.values for n in single])
FAM = {}
for n in single:
    f = json.load(open(R / f"saved_models/{n}/config.json"))["family"]
    FAM[n] = "gnn" if f.startswith("gnn") else f

META_DESC = ["MolWt", "MolLogP", "TPSA", "NumAromaticRings", "NumHDonors", "x_n_components"]
z = runner.data()["z"]; names = [str(s) for s in z["desc_names"]]
pos = pd.Series(np.arange(len(z["mol_id"])), index=z["mol_id"]).loc[dom.mol_id.values].values
META = z["desc"][pos][:, [names.index(d) for d in META_DESC]]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in dom.parent_smiles]
nn = np.zeros(len(dom))
for k in range(5):
    other = [fps[i] for i in np.where(folds != k)[0]]
    for i in np.where(folds == k)[0]:
        nn[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], other))
META = np.column_stack([META, nn])

def rmse_cols(P, y): return np.sqrt(((P - y[:, None]) ** 2).mean(0))
def family_pool(P, y):
    r = rmse_cols(P, y)
    return [min([i for i, n in enumerate(single) if FAM[n] == f], key=lambda i: r[i]) for f in sorted(set(FAM.values()))]

def fit_nnls(Ptr, ytr):
    pm, ym = Ptr.mean(0), ytr.mean(); w, _ = nnls(Ptr - pm, ytr - ym); return w, float(ym - w @ pm)

class NNLSBlend:
    def fit(self, rows):
        self.w, self.b = fit_nnls(P[rows], y[rows]); return self
    def predict(self, rows): return P[rows] @ self.w + self.b
    def export(self): return dict(components=[dict(folder=n, weight=float(w)) for n, w in zip(single, self.w) if w > 1e-6], intercept=self.b)

class RidgeStack:
    def fit(self, rows):
        self.pool = family_pool(P[rows], y[rows])
        X = P[rows][:, self.pool]; yy = y[rows]; f = folds[rows]
        best = None
        for a in [0.1, 1, 10, 100, 1000]:          # inner CV over the 4 training folds
            e = []
            for k in np.unique(f):
                m = Ridge(alpha=a).fit(X[f != k], yy[f != k]); e.append(((m.predict(X[f == k]) - yy[f == k]) ** 2).sum())
            if best is None or sum(e) < best[0]: best = (sum(e), a)
        self.alpha = best[1]; self.m = Ridge(alpha=self.alpha).fit(X, yy); return self
    def predict(self, rows): return self.m.predict(P[rows][:, self.pool])

LGB_STACK = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=2, verbose=-1,
                 deterministic=True, force_row_wise=True)
class LGBMStack:
    def fit(self, rows):
        import lightgbm as lgb
        self.pool = family_pool(P[rows], y[rows])
        self.m = lgb.LGBMRegressor(**LGB_STACK).fit(np.column_stack([P[rows][:, self.pool], META[rows]]), y[rows]); return self
    def predict(self, rows): return self.m.predict(np.column_stack([P[rows][:, self.pool], META[rows]]))

def nested(cls):
    oof = np.zeros(len(y)); per = []
    for k in range(5):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        oof[va] = cls().fit(tr).predict(va); per.append(runner.metrics(y[va], oof[va]))
    return oof, per

def save(name, kind, final, oof, per, why):
    folder = R / "saved_models" / name
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for f in ["predict_template.py:predict.py", "featurize.py:featurize.py", "stack_predict.py:stack_predict.py"]:
        a, b = f.split(":"); shutil.copy(R / "src" / a, folder / b)
    pooled = runner.metrics(y, oof)
    if kind == "nnls":
        cfg = dict(name=name, family="blend", **final.export(), candidate_pool=single,
                   combination="LogS = intercept + sum_i weight_i * predict_i(SMILES)")
        comps = [c["folder"] for c in cfg["components"]]
    else:
        comps = [single[i] for i in final.pool]
        cfg = dict(name=name, family="stack", stacker=kind, components=comps, candidate_pool=single,
                   meta_desc=META_DESC if kind == "lgbm" else None, meta_nn_sim=kind == "lgbm",
                   desc_names=None, stacker_params=LGB_STACK if kind == "lgbm" else dict(alpha=final.alpha),
                   feature_order="component predictions in 'components' order" + (", then meta_desc, then nn_sim" if kind == "lgbm" else ""))
        if kind == "ridge":
            import joblib; joblib.dump(final.m, folder / "stacker.joblib")
            cfg["ridge_coef"] = [float(c) for c in final.m.coef_]; cfg["ridge_intercept"] = float(final.m.intercept_)
        else:
            final.m.booster_.save_model(str(folder / "stacker.lgbm.txt"))
            shutil.copy(R / "saved_models/train_parent_fps.npz", folder / "train_parent_fps.npz")
    cfg.update(input="SMILES string(s)", output="predicted LogS (log10 mol/L)", note=why,
               cv=dict(per_fold=per, pooled=pooled, nested=True), training_script="scripts/r2_06_stack.py",
               trained_on="OOF predictions of in-domain training molecules (data/06b_train_v2.csv)")
    json.dump(cfg, open(folder / "config.json", "w"), indent=2)
    # reference predictions through the saved loader itself; the independent check is test-metric reproduction
    import importlib.util
    ref = pd.read_csv(R / "saved_models/lgbm_desc/reference_predictions.csv")
    spec = importlib.util.spec_from_file_location(f"p_{name}", folder / "predict.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    pd.DataFrame(dict(smiles=ref.smiles, pred_at_save=mod.load_predictor(folder)(ref.smiles.tolist()))).to_csv(folder / "reference_predictions.csv", index=False)
    pd.DataFrame(dict(mol_id=dom.mol_id, cv_fold=folds, y=y, oof=oof)).to_csv(R / f"results/oof/{name}.csv", index=False)
    row = dict(name=name, family=cfg["family"], feature_set="(components)", params=json.dumps({k: cfg[k] for k in cfg if k in ("components", "intercept", "stacker", "stacker_params")}),
               seed=42, cv_rmse_pooled=pooled["rmse"], cv_r2_pooled=pooled["r2"], cv_mae_pooled=pooled["mae"],
               cv_rmse_fold_mean=float(np.mean([p["rmse"] for p in per])), cv_rmse_fold_sd=float(np.std([p["rmse"] for p in per], ddof=1)),
               cv_r2_fold_mean=float(np.mean([p["r2"] for p in per])), cv_r2_fold_sd=float(np.std([p["r2"] for p in per], ddof=1)), note=why)
    runner.log_result(row)
    log = R / "results/r2_log.csv"
    rec = pd.DataFrame([dict(name=name, family=cfg["family"], feature_set="(components)", params=row["params"][:300], why=why,
                             cv_rmse_indomain=pooled["rmse"], cv_r2_indomain=pooled["r2"], cv_mae_indomain=pooled["mae"],
                             cv_rmse_fold_sd_own=row["cv_rmse_fold_sd"], secs=0)])
    if log.exists(): old = pd.read_csv(log); rec = pd.concat([old[old.name != name], rec])
    rec.to_csv(log, index=False)
    print(f"{name}: nested CV RMSE {pooled['rmse']:.4f} R2 {pooled['r2']:.4f}  components={comps}", flush=True)

all_rows = np.arange(len(y))
print("pool:", single)
for name, cls, kind, why in [
        ("r2_blend_nnls", NNLSBlend, "nnls", "Round-1 NNLS method over the enlarged round-1 + round-2 pool."),
        ("r2_stack_ridge", RidgeStack, "ridge", "Learned (unconstrained, regularized) linear stacker over the per-family best models."),
        ("r2_stack_lgbm_meta", LGBMStack, "lgbm", "Non-linear stacker that can weight models differently by molecule size, polarity, "
                                                    "aromaticity, salt status and novelty (error analysis: errors depend strongly on these).")]:
    oof, per = nested(cls)
    save(name, kind, cls().fit(all_rows), oof, per, why)
