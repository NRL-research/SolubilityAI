"""Step 06: ensembles (fitted/scored on in-domain train molecules, see below) built ONLY from cross-validated out-of-fold (OOF) predictions.

Blend CV scores are NESTED: for each outer fold k, weights (or top-k membership) are chosen
using the OOF predictions of the other 4 folds only, then applied to fold k.  Final weights are
fitted on all OOF predictions and saved as an explicit config.
  blend_nnls      : non-negative least squares weights over all saved single models (+ intercept)
  blend_mean_top3 : equal-weight mean of the 3 best single models by OOF RMSE
  blend_nnls_family: NNLS over a pool of one model per family (best OOF RMSE within each family)
Components = every saved single model with an OOF file (list recorded in config).
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner

res = pd.read_csv(R / "results/cv_results.csv")
single = [n for n in res.name if (R / f"results/oof/{n}.csv").exists() and not n.startswith("blend")
          and (R / f"saved_models/{n}/config.json").exists()]
# All blends are fitted and scored on the IN-DOMAIN training molecules (dataset v2 train, 01b),
# where every candidate (v1- or v2-trained) has an out-of-fold prediction from the same folds.
dom = pd.read_csv(R / "data/06b_train_v2.csv")[["mol_id", "cv_fold"]]
O = {n: dom[["mol_id"]].merge(pd.read_csv(R / f"results/oof/{n}.csv"), on="mol_id", how="left") for n in single}
base = O[single[0]]
for n in single:
    assert (O[n].mol_id.values == base.mol_id.values).all() and O[n].oof.notna().all(), n
    assert (O[n].cv_fold.values == dom.cv_fold.values).all()
y, folds = base.y.values, base.cv_fold.values
P = np.column_stack([O[n].oof.values for n in single])

def fit_nnls(P, y):
    pm, ym = P.mean(0), y.mean()
    w, _ = nnls(P - pm, y - ym)
    return w, float(ym - w @ pm)

def top3(P, y):
    r = np.sqrt(((P - y[:, None]) ** 2).mean(0))
    idx = np.argsort(r)[:3]; w = np.zeros(P.shape[1]); w[idx] = 1 / 3
    return w, 0.0

FAM = {n: json.load(open(R / f"saved_models/{n}/config.json"))["family"] for n in single}
def family_best_nnls(P, y):
    """Pool = the single best model (lowest RMSE on the given OOF rows) of each model family,
    then NNLS over that pool.  Chosen inside each outer fold -> fully nested."""
    r = np.sqrt(((P - y[:, None]) ** 2).mean(0))
    pool = []
    for fam in sorted(set(FAM.values())):
        cand = [i for i, n in enumerate(single) if FAM[n] == fam]
        pool.append(min(cand, key=lambda i: r[i]))
    w_sub, b = fit_nnls(P[:, pool], y)
    w = np.zeros(P.shape[1]); w[pool] = w_sub
    return w, b

out_rows = []
for bname, fitter in [("blend_nnls", fit_nnls), ("blend_mean_top3", top3), ("blend_nnls_family", family_best_nnls)]:
    oof = np.zeros(len(y)); per = []
    for k in range(5):
        tr, va = folds != k, folds == k
        w, b = fitter(P[tr], y[tr])
        oof[va] = P[va] @ w + b
        per.append(runner.metrics(y[va], oof[va]))
    pooled = runner.metrics(y, oof)
    w, b = fitter(P, y)
    comps = [dict(folder=n, weight=float(wi)) for n, wi in zip(single, w) if wi > 1e-6]
    folder = R / "saved_models" / bname
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)
    cfg = dict(name=bname, family="blend", components=comps, intercept=b,
               combination="LogS = intercept + sum_i weight_i * predict_i(SMILES), predict_i = saved_models/<folder>/predict.py",
               candidate_pool=single, weight_fitting=fitter.__name__ + " on all train OOF predictions",
               input="SMILES string(s)", output="predicted LogS (log10 mol/L)",
               cv=dict(per_fold=per, pooled=pooled, nested=True), training_script="scripts/06_blend.py")
    json.dump(cfg, open(folder / "config.json", "w"), indent=2)
    shutil.copy(R / "src/predict_template.py", folder / "predict.py")
    shutil.copy(R / "src/featurize.py", folder / "featurize.py")
    # reference predictions: run each component through its own saved predict.py on the v1
    # reference molecules (components trained on v1/v2 have different reference lists)
    import importlib.util
    ref = pd.read_csv(R / "saved_models/lgbm_desc/reference_predictions.csv")
    rp = np.full(len(ref), b)
    for c in comps:
        cf = R / "saved_models" / c["folder"]
        spec = importlib.util.spec_from_file_location(f"pc_{c['folder']}", cf / "predict.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        rp = rp + c["weight"] * mod.load_predictor(cf)(ref.smiles.tolist())
    pd.DataFrame(dict(smiles=ref.smiles, pred_at_save=rp)).to_csv(folder / "reference_predictions.csv", index=False)
    pd.DataFrame(dict(mol_id=base.mol_id, cv_fold=folds, y=y, oof=oof)).to_csv(R / f"results/oof/{bname}.csv", index=False)
    row = dict(name=bname, family="blend", feature_set="(components)", params=json.dumps(dict(components=comps, intercept=b)),
               seed=-1, cv_rmse_pooled=pooled["rmse"], cv_r2_pooled=pooled["r2"], cv_mae_pooled=pooled["mae"],
               cv_rmse_fold_mean=float(np.mean([p["rmse"] for p in per])), cv_rmse_fold_sd=float(np.std([p["rmse"] for p in per], ddof=1)),
               cv_r2_fold_mean=float(np.mean([p["r2"] for p in per])), cv_r2_fold_sd=float(np.std([p["r2"] for p in per], ddof=1)),
               note="nested-CV blend over OOF predictions")
    runner.log_result(row); print(json.dumps(row)); print(comps)
