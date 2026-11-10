"""
23_stacking.py
==============
Stacked ensembling: a meta-model that LEARNS how to combine base-model predictions,
instead of the fixed non-negative weights used so far.

Three levels of ambition, all evaluated under the same 5 scaffold-grouped folds:

  1. `blend_nnls`      the existing approach, for reference: one non-negative weight
                       per base model, applied uniformly to every molecule.
  2. `stack_ridge`     Ridge on base predictions. Can use negative coefficients and an
                       intercept, so it can correct systematic bias that non-negative
                       weights cannot -- and the error analysis found real bias:
                       +0.90 LogS in the least-soluble decile, -0.67 in the most
                       soluble. Shrinkage at both tails is exactly what a linear
                       correction with an intercept and free signs can attack.
  3. `stack_gbm`       LightGBM on base predictions. Weighting becomes a function of
                       the predictions themselves, so the ensemble can behave
                       differently in different regions of LogS -- a non-linear
                       recalibration, which subsumes isotonic correction of the tails.
  4. `stack_*_meta`    same, plus META-FEATURES: a handful of chem_aware columns
                       (fragment count, inorganic flag, charge, largest-fragment
                       size/logP). Motivated directly by the error analysis: inorganic
                       species (9.5% of molecules, 27.2% of squared error) and salts
                       (11.2%, 25.3%) fail differently from ordinary organics, so a
                       meta-model that can SEE "this is a metal salt" may route it to
                       whichever base model handles that class best, rather than
                       applying one global weighting to everything.

Honesty note on the CV numbers here
-----------------------------------
The meta-model is cross-fitted over the SAME 5 folds: for fold k it is trained on the
OOF rows of the other folds and predicts fold k. That is standard stacking practice,
but it carries a known mild optimism -- the base OOF values used as meta-training
features for folds != k came from base models whose training data included fold k's
rows. Removing that entirely needs nested CV (inner folds inside every outer fold),
which costs 5x the entire base-model sweep and is out of reach on this hardware.

So the CV numbers below are used ONLY to choose between stackers. The number that
decides whether stacking actually beat the weighted blend is the single held-out test
evaluation in 24_test_eval_round3.py, where the meta-model is fit on train+val OOF
once and applied unchanged to base predictions from models trained on full train+val.
That path has no test leakage of any kind.

Output: results/stacking_cv.csv, results/stacking_models.json
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import optimize_blend_weights  # noqa: E402

DATA = "/home/claude/work/data/v2_corrected"
CACHE = "/home/claude/work/data/features_cache"
RES = "/home/claude/work/results"
OOF = f"{RES}/oof"

trainval = pd.read_csv(f"{DATA}/trainval.csv")
folds = json.load(open(f"{DATA}/cv_folds.json"))["folds"]
y = trainval["Solubility"].values
n = len(y)

# meta-features: the error-analysis-motivated subset of chem_aware
META_COLS = ["n_fragments", "is_inorganic", "has_metal", "is_charged",
             "net_formal_charge", "n_heavy_largest", "largest_frag_heavy_fraction",
             "lf_MolWt", "lf_MolLogP", "lf_TPSA", "carbon_fraction", "n_metal_atoms"]
reg = json.load(open("/home/claude/work/data/feature_name_registry.json"))
ca_names = reg["chem_aware"]
ca_idx = [ca_names.index(c) for c in META_COLS]
CA = np.vstack([np.load(f"{CACHE}/train__chem_aware.npy"),
                np.load(f"{CACHE}/val__chem_aware.npy")]).astype(np.float64)[:, ca_idx]
assert CA.shape[0] == n


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    return float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))


# ---------------------------------------------------------------- load base OOF
preds, meta_info = {}, {}
for f in sorted(glob.glob(f"{OOF}/*.npz")):
    d = np.load(f, allow_pickle=True)
    name = os.path.basename(f)[:-4]
    m = json.loads(str(d["meta"]))
    preds[name] = d["oof"]
    meta_info[name] = m

gnn = sorted(glob.glob(f"{RES}/gnn/oof_fold*_seed42.npz"))
if len(gnn) == len(folds):
    o = np.full(n, np.nan)
    for f in gnn:
        d = np.load(f)
        o[d["idx"]] = d["pred"]
    assert not np.isnan(o).any()
    preds["gatv2[graph+5desc]"] = o
    meta_info["gatv2[graph+5desc]"] = {"pooled_oof_rmse": rmse(y, o),
                                       "model_type": "gatv2",
                                       "feature_set": "graph+5desc"}

ranked = sorted(preds, key=lambda k: rmse(y, preds[k]))
print(f"{len(preds)} base models available. Best 12 by pooled-OOF RMSE:")
for k in ranked[:12]:
    print(f"  {rmse(y, preds[k]):.4f}  {k}")

# base-model sets to stack over
STRONG = [k for k in ranked if rmse(y, preds[k]) < 1.10][:10]
DIVERSE = []
seen_m, seen_f = set(), set()
for k in ranked:
    mt, fs = meta_info[k].get("model_type"), meta_info[k].get("feature_set")
    if mt in seen_m and fs in seen_f:
        continue
    DIVERSE.append(k)
    seen_m.add(mt)
    seen_f.add(fs)
    if len(DIVERSE) == 10:
        break
ALL_USABLE = [k for k in ranked if rmse(y, preds[k]) < 1.45]

ALL_MODELS = list(ranked)          # every base model, no RMSE threshold at all
SETS = {"strong10": STRONG, "diverse10": DIVERSE, "all_usable": ALL_USABLE,
        "all_models": ALL_MODELS}
for nm, s in SETS.items():
    print(f"\n{nm} ({len(s)}): {s}")


# ---------------------------------------------------------------- stackers
def fit_predict(kind, P_tr, y_tr, M_tr, P_te, M_te, seed=42):
    """Fit a meta-model on (base predictions [, meta-features]) and predict."""
    use_meta = kind.endswith("_meta")
    Xtr = np.hstack([P_tr, M_tr]) if use_meta else P_tr
    Xte = np.hstack([P_te, M_te]) if use_meta else P_te
    base = kind[:-5] if use_meta else kind

    if base == "blend_nnls":
        w = optimize_blend_weights(P_tr, y_tr, seed=seed)
        return P_te @ w, {"weights": [float(x) for x in w]}
    if base == "stack_ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        m = Pipeline([("s", StandardScaler()),
                      ("m", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
        m.fit(Xtr, y_tr)
        return m.predict(Xte), {"alpha": float(m.named_steps["m"].alpha_)}
    if base == "stack_enet":
        from sklearn.linear_model import ElasticNetCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        m = Pipeline([("s", StandardScaler()),
                      ("m", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], n_alphas=30,
                                         cv=4, max_iter=5000, random_state=seed,
                                         n_jobs=-1))])
        m.fit(Xtr, y_tr)
        return m.predict(Xte), {"alpha": float(m.named_steps["m"].alpha_),
                                "l1_ratio": float(m.named_steps["m"].l1_ratio_)}
    if base == "stack_gbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                              max_depth=4, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0,
                              min_child_samples=40, random_state=seed,
                              n_jobs=-1, verbose=-1)
        m.fit(Xtr, y_tr)
        return m.predict(Xte), {}
    raise ValueError(kind)


KINDS = ["blend_nnls", "stack_ridge", "stack_ridge_meta", "stack_enet",
         "stack_gbm", "stack_gbm_meta"]

rows = []
for set_name, members in SETS.items():
    P = np.column_stack([preds[m] for m in members])
    for kind in KINDS:
        # cross-fitted over the same folds
        oof_stack = np.full(n, np.nan)
        for f in folds:
            tr, va = np.array(f["train"]), np.array(f["val"])
            p, _ = fit_predict(kind, P[tr], y[tr], CA[tr], P[va], CA[va])
            oof_stack[va] = p
        assert not np.isnan(oof_stack).any()
        rows.append({"base_set": set_name, "n_members": len(members), "stacker": kind,
                     "cv_rmse": rmse(y, oof_stack), "cv_r2": r2(y, oof_stack),
                     "cv_mae": float(np.mean(np.abs(y - oof_stack)))})
        print(f"{set_name:11s} {kind:18s} cross-fitted CV RMSE={rows[-1]['cv_rmse']:.4f}  "
              f"R2={rows[-1]['cv_r2']:.4f}", flush=True)

df = pd.DataFrame(rows).sort_values("cv_rmse")
df.to_csv(f"{RES}/stacking_cv.csv", index=False)

best = df.iloc[0]
print(f"\nBest stacker by cross-fitted CV: {best['stacker']} on {best['base_set']} "
      f"(RMSE {best['cv_rmse']:.4f})")
print(f"Current best single model: lightgbm[mordred] {rmse(y, preds['lightgbm[mordred]']):.4f}")

# freeze the configuration(s) to be evaluated on test, before test is read
chosen = {}
for _, r in df.head(3).iterrows():
    key = f"{r['stacker']}@{r['base_set']}"
    chosen[key] = {"stacker": r["stacker"], "base_set": r["base_set"],
                   "members": SETS[r["base_set"]], "cv_rmse": float(r["cv_rmse"]),
                   "meta_cols": META_COLS if r["stacker"].endswith("_meta") else []}
json.dump({"sets": SETS, "chosen_for_test": chosen,
           "meta_cols": META_COLS,
           "note": "chosen by cross-fitted CV on train+val OOF only; test not read"},
          open(f"{RES}/stacking_models.json", "w"), indent=2)
print(f"\nSaved {RES}/stacking_cv.csv and stacking_models.json")
print(df.to_string(index=False))
