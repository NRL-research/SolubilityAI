"""
24_test_eval_round3.py
======================
The single held-out test evaluation for round 3.

Everything evaluated here was chosen from out-of-fold train+val predictions only
(23_stacking.py -> results/stacking_models.json, written before test was opened).
This script:

  1. Trains every base model needed by the frozen stack configurations on the FULL
     train+val pool, once, and predicts the test set once. Cached to
     results/test_preds_round3/ so it is resumable and so no model is fit twice.
  2. Fits each meta-model (Ridge / NNLS) on the train+val OOF matrix -- never on test
     -- and applies it unchanged to the test predictions.
  3. Reports test RMSE / MAE / R2 for each, alongside the existing champions.

Per-model seeds match the round-2 protocol: 3 seeds for tree/sklearn models (42, 101,
123), 5 for GATv2, seed-averaged. GATv2 test predictions are reused from
results/gnn/, which were produced by the same protocol.

Output: results/test_preds_round3/*.npy, results/final_test_results_round3.csv,
        results/test_predictions_round3.npz
"""

import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_eval import metrics  # noqa: E402
from models import build, optimize_blend_weights  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "nmf", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "22_new_model_families.py"))
nmf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nmf)

DATA = "/home/claude/work/data/v2_corrected"
CACHE = "/home/claude/work/data/features_cache"
RES = "/home/claude/work/results"
OOF = f"{RES}/oof"
TP = f"{RES}/test_preds_round3"
os.makedirs(TP, exist_ok=True)

TREE_SEEDS = [42, 101, 123]

trainval = pd.read_csv(f"{DATA}/trainval.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y_tv = trainval["Solubility"].values
y_te = test["Solubility"].values
n_train = len(pd.read_csv(f"{DATA}/train.csv"))

cfg = json.load(open(f"{RES}/stacking_models.json"))
chosen = cfg["chosen_for_test"]
META_COLS = cfg["meta_cols"]

reg = json.load(open("/home/claude/work/data/feature_name_registry.json"))
ca_names = reg["chem_aware"]
ca_idx = [ca_names.index(c) for c in META_COLS]
CA_tv = np.vstack([np.load(f"{CACHE}/train__chem_aware.npy"),
                   np.load(f"{CACHE}/val__chem_aware.npy")]).astype(np.float64)[:, ca_idx]
CA_te = np.load(f"{CACHE}/test__chem_aware.npy").astype(np.float64)[:, ca_idx]


def load_X(fs):
    tr = np.load(f"{CACHE}/train__{fs}.npy")
    va = np.load(f"{CACHE}/val__{fs}.npy")
    te = np.load(f"{CACHE}/test__{fs}.npy")
    X_tv = np.vstack([tr, va]).astype(np.float64)
    assert X_tv.shape[0] == len(trainval) and tr.shape[0] == n_train
    return X_tv, te.astype(np.float64)


def parse(config):
    """'lightgbm[mordred]' -> ('lightgbm', 'mordred')"""
    name, _, rest = config.partition("[")
    return name, rest.rstrip("]")


ROUND2_ALIASES = {"catboost_solo": "catboost", "xgboost_branch": "xgboost"}


def test_pred_for(config):
    """Seed-averaged test prediction for one base model, cached on disk."""
    out = f"{TP}/{config}.npy"
    if os.path.exists(out):
        return np.load(out)

    if config == "gatv2[graph+5desc]":
        files = sorted(glob.glob(f"{RES}/gnn/test_pred_seed*.npz"))
        assert files, "GATv2 test predictions missing"
        p = np.column_stack([np.load(f)["pred"] for f in files]).mean(axis=1)
        np.save(out, p)
        return p

    name, fs = parse(config)
    X_tv, X_te = load_X(fs)
    t0 = time.time()
    # Route by which factory knows the name. nmf.make() covers every round-3 family
    # (including the variants added after nmf.ALL was last written, so membership of
    # that list is not a reliable test); models.py build() covers round 2.
    try:
        nmf.make(name, seed=42)
        factory = "round3"
    except ValueError:
        factory = "round2"

    preds = []
    for s in TREE_SEEDS:
        if factory == "round3":
            est, _ = nmf.make(name, seed=s)
            est.fit(X_tv, y_tv)
            preds.append(np.asarray(est.predict(X_te)).ravel())
        else:
            mt = ROUND2_ALIASES.get(name, name)
            m = build(mt, seed=s)
            m.fit(X_tv, y_tv)
            preds.append(np.asarray(m.predict(X_te)).ravel())
    p = np.column_stack(preds).mean(axis=1)
    np.save(out, p)
    print(f"  fitted {config} ({time.time()-t0:.0f}s)", flush=True)
    return p


# ---------------------------------------------------------------- base OOF (for meta fit)
oof = {}
for f in sorted(glob.glob(f"{OOF}/*.npz")):
    d = np.load(f, allow_pickle=True)
    oof[os.path.basename(f)[:-4]] = d["oof"]
gnn = sorted(glob.glob(f"{RES}/gnn/oof_fold*_seed42.npz"))
if len(gnn) == 5:
    o = np.full(len(trainval), np.nan)
    for f in gnn:
        d = np.load(f)
        o[d["idx"]] = d["pred"]
    oof["gatv2[graph+5desc]"] = o

needed = sorted({m for c in chosen.values() for m in c["members"]})
print(f"{len(needed)} base models needed across {len(chosen)} frozen stack configs")
missing_oof = [m for m in needed if m not in oof]
assert not missing_oof, f"no OOF for {missing_oof}"

print("\nProducing test predictions (train on full train+val, predict test once):")
test_base = {m: test_pred_for(m) for m in needed}

# ---------------------------------------------------------------- apply the stackers
rows, preds_out = [], {}

for key, c in chosen.items():
    members = c["members"]
    P_tv = np.column_stack([oof[m] for m in members])
    P_te = np.column_stack([test_base[m] for m in members])
    kind = c["stacker"]
    use_meta = kind.endswith("_meta")
    Xtv = np.hstack([P_tv, CA_tv]) if use_meta else P_tv
    Xte = np.hstack([P_te, CA_te]) if use_meta else P_te
    base = kind[:-5] if use_meta else kind

    if base == "blend_nnls":
        w = optimize_blend_weights(P_tv, y_tv, seed=42)
        p = P_te @ w
        detail = {"weights": dict(zip(members, np.round(w, 5).tolist()))}
    elif base == "stack_ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        m = Pipeline([("s", StandardScaler()),
                      ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
        m.fit(Xtv, y_tv)
        p = m.predict(Xte)
        coefs = m.named_steps["m"].coef_
        detail = {"alpha": float(m.named_steps["m"].alpha_),
                  "intercept": float(m.named_steps["m"].intercept_),
                  "coef": dict(zip(members + (META_COLS if use_meta else []),
                                   np.round(coefs, 5).tolist()))}
    elif base == "stack_enet":
        from sklearn.linear_model import ElasticNetCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        m = Pipeline([("s", StandardScaler()),
                      ("m", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=30,
                                         cv=4, max_iter=5000, random_state=42,
                                         n_jobs=-1))])
        m.fit(Xtv, y_tv)
        p = m.predict(Xte)
        en = m.named_steps["m"]
        coefs = en.coef_
        detail = {"alpha": float(en.alpha_), "l1_ratio": float(en.l1_ratio_),
                  "intercept": float(en.intercept_),
                  "n_nonzero_members": int(np.sum(coefs != 0)),
                  "coef": dict(zip(members + (META_COLS if use_meta else []),
                                   np.round(coefs, 5).tolist()))}
    elif base == "stack_gbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                              max_depth=4, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0,
                              min_child_samples=40, random_state=42,
                              n_jobs=-1, verbose=-1)
        m.fit(Xtv, y_tv)
        p = m.predict(Xte)
        detail = {}
    else:
        raise ValueError(kind)

    preds_out[key] = p
    rows.append({"config": key, "stacker": kind, "base_set": c["base_set"],
                 "n_members": len(members), "cv_rmse": c["cv_rmse"],
                 **metrics(y_te, p)})
    print(f"\n{key}: test RMSE={rows[-1]['rmse']:.4f} R2={rows[-1]['r2']:.4f} "
          f"(cross-fitted CV was {c['cv_rmse']:.4f})")
    json.dump(detail, open(f"{RES}/stacker_detail_{key.replace('@','_at_')}.json", "w"),
              indent=2, default=str)

# individual base models, for the record
for m in needed:
    rows.append({"config": m, "stacker": "-", "base_set": "-", "n_members": 1,
                 "cv_rmse": float(np.sqrt(np.mean((y_tv - oof[m]) ** 2))),
                 **metrics(y_te, test_base[m])})
    preds_out[m] = test_base[m]

# carry the round-2 champions through unchanged for comparison
r2prev = np.load(f"{RES}/test_predictions.npz")
for champ in ["greedy_blend", "diverse_blend_4", "lightgbm[mordred]"]:
    if champ in r2prev.files and champ not in preds_out:
        preds_out[champ] = r2prev[champ]
        rows.append({"config": f"{champ} (round 2)", "stacker": "-", "base_set": "-",
                     "n_members": 1, "cv_rmse": None, **metrics(y_te, r2prev[champ])})

df = pd.DataFrame(rows).sort_values("rmse")
df.to_csv(f"{RES}/final_test_results_round3.csv", index=False)
np.savez(f"{RES}/test_predictions_round3.npz", y=y_te, **preds_out)
pd.set_option("display.width", 220)
print("\n" + df.to_string(index=False))
print(f"\nSaved {RES}/final_test_results_round3.csv")
