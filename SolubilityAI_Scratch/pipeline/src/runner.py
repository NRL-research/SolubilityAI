"""CV + final-fit + save-as-you-go runner for tabular configurations.

For a configuration (name, family, feature_set, params, seed):
  1. 5-fold grouped CV on TRAIN ONLY (folds from data/05_split_assignments.csv) -> OOF preds
  2. refit on all of TRAIN with the same params/seed
  3. immediately save to saved_models/<name>/ (native model file, preprocessor, featurize.py,
     predict.py, config.json, reference_predictions.csv)
The test set is never loaded here.
"""
import json, shutil, time, platform
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parents[1]
SRC = R / "src"
import zoo  # noqa

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(rmse=rmse, mae=float(np.mean(np.abs(y - p))), r2=r2)

import os
DATASET = os.environ.get("SOL_DATA", "v1")   # v1 = full curated; v2 = applicability-domain subset (01b)
SPLIT_FILE = {"v1": "data/05_split_assignments.csv", "v2": "data/05b_split_assignments_v2.csv"}[DATASET]
TRAIN_FILE = {"v1": "data/06_train.csv", "v2": "data/06b_train_v2.csv"}[DATASET]
_DATA = {}
def data():
    if not _DATA:
        df = pd.read_csv(R / SPLIT_FILE)
        z = np.load(R / "data/07_features.npz", allow_pickle=False)
        pos = pd.Series(np.arange(len(z["mol_id"])), index=z["mol_id"]).loc[df.mol_id.values].values
        zz = {k: (z[k][pos] if z[k].shape[0] == len(z["mol_id"]) and k != "desc_names" else z[k]) for k in z.files}
        assert (zz["mol_id"] == df.mol_id.values).all()
        _DATA.update(df=df, z=zz)
    return _DATA

def _mordred():
    if "mordred" not in _DATA:
        m = np.load(R / "data/07b_features_mordred.npz", allow_pickle=False)
        pos = pd.Series(np.arange(len(m["mol_id"])), index=m["mol_id"]).loc[data()["df"].mol_id.values].values
        _DATA["mordred"] = m["mordred"][pos]; _DATA["mordred_names"] = [str(x) for x in m["mordred_names"]]
    return _DATA["mordred"]

def _r2():
    if "r2" not in _DATA:
        m = np.load(R / "data/07c_features_r2.npz", allow_pickle=False)
        pos = pd.Series(np.arange(len(m["mol_id"])), index=m["mol_id"]).loc[data()["df"].mol_id.values].values
        _DATA["r2"] = {k: m[k][pos] for k in m.files if k != "mol_id"}
        if (R / "data/07d_features_erg2.npz").exists():      # round-2 bug-fix block (deterministic ErG)
            m2 = np.load(R / "data/07d_features_erg2.npz", allow_pickle=False)
            p2 = pd.Series(np.arange(len(m2["mol_id"])), index=m2["mol_id"]).loc[data()["df"].mol_id.values].values
            _DATA["r2"]["erg2"] = m2["erg2"][p2]
    return _DATA["r2"]

def features(feature_set, idx):
    z = data()["z"]
    if feature_set.startswith("r2:"):          # round-2 '+'-joined blocks (see featurize.featurize_r2)
        out = []
        for b in feature_set[3:].split("+"):
            if b == "desc": out.append(z["desc"][idx])
            elif b == "mordred": out.append(_mordred()[idx])
            elif b == "morgan": out.append(z["morgan_counts"][idx])
            elif b == "maccs": out.append(z["maccs"][idx])
            else: out.append(_r2()[b][idx])
        return np.hstack([np.asarray(x, dtype=np.float64) for x in out])
    if feature_set == "mordred":
        return _mordred()[idx].astype(np.float64)
    if feature_set == "desc+mordred":
        return np.hstack([z["desc"][idx].astype(np.float64), _mordred()[idx].astype(np.float64)])
    blocks = {"rdkit_desc": ["desc"], "morgan_counts_r2_2048": ["morgan_counts"], "maccs": ["maccs"],
              "esol4": ["esol4"], "desc+morgan_counts_r2_2048": ["desc", "morgan_counts"],
              "desc+maccs": ["desc", "maccs"], "morgan_bits_r2_2048": ["morgan_counts"]}[feature_set]
    X = np.hstack([z[b][idx].astype(np.float64) for b in blocks])
    if feature_set == "morgan_bits_r2_2048":
        X = (X > 0).astype(np.float64)
    return X

def versions():
    import sklearn, rdkit, lightgbm, xgboost, catboost
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, rdkit=rdkit.__version__, lightgbm=lightgbm.__version__,
                xgboost=xgboost.__version__, catboost=catboost.__version__)

def reference_smiles():
    df = data()["df"]
    tr = df[df.split == "train"].sort_values("mol_id")
    return tr.SMILES.values[:: len(tr) // 20][:20].tolist()

def cv(family, feature_set, params, seed):
    df = data()["df"]
    tr = np.where(df.split == "train")[0]
    folds = df.cv_fold.values[tr]
    y = df.LogS.values[tr]
    X = features(feature_set, tr)
    oof = np.full(len(tr), np.nan); per = []
    for k in range(5):
        va, fi = folds == k, folds != k
        f = zoo.fit(family, params, seed, X[fi], y[fi])
        oof[va] = f.predict(X[va])
        per.append(metrics(y[va], oof[va]))
    pooled = metrics(y, oof)
    return oof, per, pooled, X, y, tr

def run(name, family, feature_set, params, seed=42, note="", save_model=True, hp_search_log=None):
    t0 = time.time()
    oof, per, pooled, X, y, tr = cv(family, feature_set, params, seed)
    df = data()["df"]
    out = dict(name=name, family=family, feature_set=feature_set, params=json.dumps(params), seed=seed,
               cv_rmse_pooled=pooled["rmse"], cv_r2_pooled=pooled["r2"], cv_mae_pooled=pooled["mae"],
               cv_rmse_fold_mean=float(np.mean([p["rmse"] for p in per])),
               cv_rmse_fold_sd=float(np.std([p["rmse"] for p in per], ddof=1)),
               cv_r2_fold_mean=float(np.mean([p["r2"] for p in per])),
               cv_r2_fold_sd=float(np.std([p["r2"] for p in per], ddof=1)), note=note)
    if not save_model:
        out["secs"] = time.time() - t0
        return out, oof
    (R / "results/oof").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(mol_id=df.mol_id.values[tr], cv_fold=df.cv_fold.values[tr], y=y, oof=oof)).to_csv(
        R / f"results/oof/{name}.csv", index=False)
    # ---- final fit on all train + immediate save ----
    f = zoo.fit(family, params, seed, X, y)
    folder = R / "saved_models" / name
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)
    files = zoo.save(f, folder)
    shutil.copy(SRC / "featurize.py", folder / "featurize.py")
    shutil.copy(SRC / "predict_template.py", folder / "predict.py")
    if "mol2vec" in feature_set:
        shutil.copy(SRC / "mol2vec_vectors.npz", folder / "mol2vec_vectors.npz"); files.append("mol2vec_vectors.npz")
    z = data()["z"]
    cfg = dict(name=name, family=family, feature_set=feature_set, params=params, seed=seed,
               desc_names=[str(s) for s in z["desc_names"]] if "desc" in feature_set else None,
               mordred_names=_DATA.get("mordred_names") if "mordred" in feature_set else None,
               r2_blocks=feature_set[3:].split("+") if feature_set.startswith("r2:") else None,
               input="SMILES string(s)", output="predicted LogS (log10 of solubility in mol/L)",
               trained_on="%s (n=%d), dataset %s (split file %s)" % (TRAIN_FILE, len(tr), DATASET, SPLIT_FILE), dataset=DATASET,
               cv=dict(per_fold=per, pooled=pooled), model_files=files, versions=versions(),
               training_script=hp_search_log or "scripts/04_cv_sweep.py", note=note)
    json.dump(cfg, open(folder / "config.json", "w"), indent=2)
    ref = reference_smiles()
    idx = [int(np.where(df.SMILES.values == s)[0][0]) for s in ref]
    ref_pred = f.predict(features(feature_set, idx))
    pd.DataFrame(dict(smiles=ref, pred_at_save=ref_pred)).to_csv(folder / "reference_predictions.csv", index=False)
    out["secs"] = time.time() - t0
    return out, oof

def log_result(row, path=R / "results/cv_results.csv"):
    d = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path); old = old[old.name != row["name"]]
        d = pd.concat([old, d])
    d.to_csv(path, index=False)
