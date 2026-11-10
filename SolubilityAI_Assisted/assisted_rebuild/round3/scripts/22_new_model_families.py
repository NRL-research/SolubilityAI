"""
22_new_model_families.py
========================
Model families and objectives this project has not tried, under the identical 5-fold
scaffold-grouped CV protocol (test never opened). Writes OOF predictions to
results/oof/ so they drop straight into the existing leaderboard and stacker.

Why each of these, given what the error analysis showed
-------------------------------------------------------
Robust objectives (huber, mae, quantile-median). The OOF error is extremely
  concentrated: the worst 1% of molecules carry 20.4% of the squared error and the
  worst 5% carry 47.7%. A squared-error objective spends most of its gradient chasing
  that tail, and a good part of the tail looks like AqSolDB label noise rather than
  chemistry (e.g. "Propyzamide" recorded at LogS -4.23 with the SMILES of
  propionamide, which is freely soluble). A loss that stops chasing it may fit the
  other 95% better -- which can lower overall RMSE even though RMSE is what the
  squared loss directly optimises. Empirical question, worth answering.

Scaled linear / kernel / neural models (elasticnet, svr_rbf, mlp). Never tried on
  this foundation. Individually unlikely to beat gradient boosting on tabular
  descriptors, but they make errors of a completely different shape, which is what an
  ensemble actually needs -- the existing blend members are all axis-aligned trees.
  These are the project's FIRST models needing a fitted preprocessor (StandardScaler),
  so they also exercise that path end to end.

k-nearest-neighbours on Tanimoto similarity (knn_tanimoto). The classical
  read-across baseline for solubility, and the same mechanism the project's abandoned
  RAG-LLM pipeline used for retrieval. Purely local, zero parametric assumptions --
  maximally different from a boosted tree. Included to see whether local structural
  similarity carries signal the global models miss.

Deeper / more regularised boosting variants (lightgbm_deep, lightgbm_conservative).
  The existing LightGBM config was hand-set, never searched. Two meaningfully
  different points in the space rather than a nudge.

Preprocessing note: the scaled models fit a StandardScaler INSIDE each CV fold, on
that fold's training rows only, via sklearn Pipeline -- never on the full train+val
pool. Fitting the scaler once outside the folds would leak fold-validation
distribution into fold training and inflate every CV number here.

Usage: python3 22_new_model_families.py <name>[:feature_set] ...
       python3 22_new_model_families.py --list
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = "/home/claude/work/data/v2_corrected"
CACHE = "/home/claude/work/data/features_cache"
OUT = "/home/claude/work/results/oof"
os.makedirs(OUT, exist_ok=True)

trainval = pd.read_csv(f"{DATA}/trainval.csv")
n_train = len(pd.read_csv(f"{DATA}/train.csv"))
folds = json.load(open(f"{DATA}/cv_folds.json"))["folds"]
y = trainval["Solubility"].values


def make(name, seed=42):
    """Returns (estimator, default_feature_set). Estimators that need scaling are
    wrapped in a Pipeline so the scaler is fit per fold, inside the CV."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "lightgbm_huber":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="huber", alpha=2.0, n_estimators=1500, learning_rate=0.03,
            num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
            random_state=seed, n_jobs=-1, verbose=-1), "mordred"
    if name.startswith("lightgbm_huber_d"):
        # Probe of the Huber delta knob. delta=2.0 (lightgbm_huber) beat both plain L2
        # (1.0135) and full L1 (1.0269) at pooled-OOF 1.0081, so the useful amount of
        # robustness is partial: enough to stop chasing the noisy tail the error
        # analysis found, not so much that the tail is ignored. These bracket it.
        import lightgbm as lgb
        delta = float(name.split("_d")[-1].replace("p", "."))
        return lgb.LGBMRegressor(
            objective="huber", alpha=delta, n_estimators=1500, learning_rate=0.03,
            num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
            random_state=seed, n_jobs=-1, verbose=-1), "mordred"
    if name == "lightgbm_mae":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=1500, learning_rate=0.03,
            num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
            random_state=seed, n_jobs=-1, verbose=-1), "mordred"
    if name == "lightgbm_deep":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.015, num_leaves=255, max_depth=-1,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.4,
            reg_alpha=0.5, reg_lambda=5.0, min_child_samples=10,
            random_state=seed, n_jobs=-1, verbose=-1), "mordred"
    if name == "lightgbm_conservative":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=4000, learning_rate=0.01, num_leaves=15, max_depth=6,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=10.0, min_child_samples=40,
            random_state=seed, n_jobs=-1, verbose=-1), "mordred"
    if name in ("lightgbm_cons_huber", "lightgbm_cons_huber_ca"):
        # Combines the round-3's two independent wins: heavy regularisation
        # (lightgbm_conservative, pooled-OOF 1.0048) and a Huber objective
        # (lightgbm_huber, 1.0081), both of which beat the plain L2 baseline (1.0135).
        # They attack the same problem from different sides -- the error analysis
        # showed a noisy, heavy tail, which over-fitting amplifies and a squared loss
        # chases -- so it is worth checking whether the two gains compound or overlap.
        import lightgbm as lgb
        fs = "mordred+chem_aware" if name.endswith("_ca") else "mordred"
        return lgb.LGBMRegressor(
            objective="huber", alpha=2.0, n_estimators=4000, learning_rate=0.01,
            num_leaves=15, max_depth=6, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.3, reg_alpha=1.0, reg_lambda=10.0,
            min_child_samples=40, random_state=seed, n_jobs=-1, verbose=-1), fs
    if name == "lightgbm_very_conservative":
        # One step further along the axis that just worked, to check whether 1.0048 is
        # a plateau or the slope is still going.
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=6000, learning_rate=0.008, num_leaves=7, max_depth=4,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.2,
            reg_alpha=2.0, reg_lambda=20.0, min_child_samples=60,
            random_state=seed, n_jobs=-1, verbose=-1), "mordred"
    if name == "lightgbm_conservative_ca":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=4000, learning_rate=0.01, num_leaves=15, max_depth=6,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=10.0, min_child_samples=40,
            random_state=seed, n_jobs=-1, verbose=-1), "mordred+chem_aware"
    if name == "catboost_huber":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            loss_function="Huber:delta=2.0", iterations=3000, learning_rate=0.02,
            depth=8, l2_leaf_reg=5, random_state=seed, verbose=0,
            allow_writing_files=False, thread_count=-1), "rdkit_full_eng"
    if name == "catboost_mae":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            loss_function="MAE", iterations=3000, learning_rate=0.02, depth=8,
            l2_leaf_reg=5, random_state=seed, verbose=0,
            allow_writing_files=False, thread_count=-1), "rdkit_full_eng"
    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return Pipeline([("scale", StandardScaler()),
                         ("m", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000,
                                          random_state=seed))]), "mordred"
    if name == "svr_rbf":
        from sklearn.svm import SVR
        return Pipeline([("scale", StandardScaler()),
                         ("m", SVR(kernel="rbf", C=10.0, gamma="scale",
                                   epsilon=0.1, cache_size=500))]), "rdkit_full_eng"
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        return Pipeline([("scale", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(512, 128),
                                            activation="relu", alpha=1e-3,
                                            learning_rate_init=1e-3, batch_size=128,
                                            max_iter=300, early_stopping=True,
                                            n_iter_no_change=15, random_state=seed))]), \
               "rdkit_full_eng"
    if name == "knn_tanimoto":
        return TanimotoKNN(k=10), "solubility_ai"
    raise ValueError(name)


class TanimotoKNN:
    """Similarity-weighted k-NN over binary Morgan bits, Tanimoto metric.

    Expects the `solubility_ai` feature matrix; uses only its first 2048 columns (the
    Morgan bits), ignoring the 5 appended descriptors, so the metric is a genuine
    fingerprint Tanimoto. Prediction is the similarity-weighted mean of the k nearest
    training molecules' LogS.
    """

    def __init__(self, k=10, n_bits=2048):
        self.k = k
        self.n_bits = n_bits

    def fit(self, X, y):
        self.Xb_ = (np.asarray(X)[:, :self.n_bits] > 0).astype(np.float32)
        self.y_ = np.asarray(y, dtype=np.float64)
        self.pop_ = self.Xb_.sum(axis=1)
        return self

    def predict(self, X):
        Q = (np.asarray(X)[:, :self.n_bits] > 0).astype(np.float32)
        qpop = Q.sum(axis=1)
        out = np.empty(len(Q))
        step = 256
        for s in range(0, len(Q), step):
            q = Q[s:s + step]
            inter = q @ self.Xb_.T
            union = qpop[s:s + step, None] + self.pop_[None, :] - inter
            tani = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
            idx = np.argpartition(-tani, self.k, axis=1)[:, :self.k]
            rows = np.arange(len(q))[:, None]
            w = tani[rows, idx]
            yv = self.y_[idx]
            wsum = w.sum(axis=1)
            flat = wsum < 1e-9
            out[s:s + step] = np.where(
                flat, yv.mean(axis=1),
                (w * yv).sum(axis=1) / np.maximum(wsum, 1e-9))
        return out

    def get_params(self, deep=False):
        return {"k": self.k, "n_bits": self.n_bits}


ALL = ["lightgbm_huber", "lightgbm_mae", "lightgbm_deep", "lightgbm_conservative",
       "catboost_huber", "catboost_mae", "elasticnet", "svr_rbf", "mlp",
       "knn_tanimoto", "lightgbm_huber_d1p0", "lightgbm_huber_d4p0"]


def load_X(fs):
    tr = np.load(f"{CACHE}/train__{fs}.npy")
    va = np.load(f"{CACHE}/val__{fs}.npy")
    X = np.vstack([tr, va]).astype(np.float64)
    assert X.shape[0] == len(trainval) and tr.shape[0] == n_train
    return X


def run(name, fs=None, seed=42):
    est, default_fs = make(name, seed)
    fs = fs or default_fs
    config = f"{name}[{fs}]"
    out = f"{OUT}/{config}.npz"
    if os.path.exists(out):
        print(f"skip {config} (exists)")
        return
    X = load_X(fs)
    oof = np.full(len(trainval), np.nan)
    fold_rmse = []
    t0 = time.time()
    for i, f in enumerate(folds):
        tr, va = np.array(f["train"]), np.array(f["val"])
        est, _ = make(name, seed)          # fresh estimator (and fresh scaler) per fold
        est.fit(X[tr], y[tr])
        p = np.asarray(est.predict(X[va])).ravel()
        oof[va] = p
        r = float(np.sqrt(np.mean((y[va] - p) ** 2)))
        fold_rmse.append(r)
        print(f"  {config} fold {i}: rmse={r:.4f} ({time.time()-t0:.0f}s)", flush=True)
    assert not np.isnan(oof).any()
    pooled = float(np.sqrt(np.mean((y - oof) ** 2)))
    try:
        params = {k: str(v) for k, v in est.get_params().items()}
    except Exception:
        params = {}
    np.savez(out, oof=oof, y=y, fold_rmse=np.array(fold_rmse),
             meta=json.dumps({"model_type": name, "feature_set": fs,
                              "kwargs": params, "seed": seed,
                              "mean_fold_rmse": float(np.mean(fold_rmse)),
                              "std_fold_rmse": float(np.std(fold_rmse)),
                              "pooled_oof_rmse": pooled,
                              "seconds": time.time() - t0,
                              "round": 3}))
    print(f"{config}: mean-fold RMSE={np.mean(fold_rmse):.4f}+-{np.std(fold_rmse):.4f}  "
          f"pooled-OOF RMSE={pooled:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--list"]:
        print("\n".join(ALL))
        sys.exit(0)
    for a in (args or ALL):
        name, _, fs = a.partition(":")
        run(name, fs or None)
