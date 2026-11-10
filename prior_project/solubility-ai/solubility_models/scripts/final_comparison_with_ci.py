"""
Follow-up to final_test_evaluation.py, addressing two verification-agent
recommendations before trusting the round-2 result for a paper:

1. A CLEAN rerun of the LITERAL dissolvr/ pipeline (dissolvr/featurizer.py's
   MoleculeFeaturizer + dissolvr/train.py's CatBoost config, 5 seeds), not
   the solubility_models/featurizer.py reimplementation used to produce the
   existing final_test_results.csv rows -- an audit found the reimplementation
   is missing dissolvr/featurizer.py's tautomer-canonicalization step
   (rdMolStandardize.TautomerEnumerator), causing ~7.7% of test molecules to
   get different descriptor values between the two code paths.

2. A PAIRED BOOTSTRAP confidence interval on (DISSOLVR RMSE - our weighted
   blend RMSE) on the exact same held-out test set, to check whether the
   observed ~3.4% RMSE improvement is larger than plausible finite-test-set
   sampling noise, not just a re-report of the point estimates.

DISSOLVR's "prediction" for pairing purposes is the mean of its 5 seeds'
predictions -- a natural single point estimate of the method, analogous to
how our own candidate is itself a 4-model blend.

Usage:
  python solubility_models/scripts/final_comparison_with_ci.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dissolvr"))

from common.splitting import scaffold_group_kfold
from common.evaluate import regression_metrics, format_metrics
from solubility_models.models import build_tree_model, build_sklearn_model
from solubility_models.ensemble import optimize_blend_weights, weighted_blend, StackingMetaLearner
from featurizer import MoleculeFeaturizer  # dissolvr/featurizer.py, the literal DISSOLVR featurizer

from catboost import CatBoostRegressor

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPLIT_DIR = os.path.join(ROOT, "common", "data", "processed_scaffold_fixed")
CACHE_DIR = os.path.join(ROOT, "solubility_models", "data", "features_cache")
DISSOLVR_DATA_DIR = os.path.join(ROOT, "dissolvr", "data")
N_FOLDS = 5
N_BOOTSTRAP = 10000
DISSOLVR_SEEDS = [42, 101, 123, 456, 789]

CANDIDATES = [
    dict(name="lightgbm[mordred]", family="tree", model_type="lightgbm", feature_set="mordred",
         params=dict(n_estimators=736, learning_rate=0.02210943033781764, num_leaves=180, max_depth=8,
                     subsample=0.7566553417347286, colsample_bytree=0.7544350241810107,
                     min_child_samples=59, reg_alpha=0.0008787301196251835, reg_lambda=0.00089760661834825)),
    dict(name="catboost[mordred]", family="tree", model_type="catboost", feature_set="mordred",
         params=dict(iterations=749, learning_rate=0.03231778142523447, depth=7,
                     l2_leaf_reg=8.130699985257577, bagging_temperature=2.6296309222169056,
                     random_strength=2.6996450535168885e-05)),
    dict(name="hist_gradient_boosting[rdkit_full+engineered]", family="sklearn",
         model_type="hist_gradient_boosting", feature_set="rdkit_full+engineered",
         params=dict(max_iter=794, learning_rate=0.05362853964441846, max_depth=10,
                     max_leaf_nodes=118, l2_regularization=1.6934490731313353e-08)),
    dict(name="extra_trees[rdkit_descriptors]", family="sklearn", model_type="extra_trees",
         feature_set="rdkit_descriptors",
         params=dict(n_estimators=104, max_depth=30, min_samples_leaf=7, max_features=0.9461903383828855)),
]

_CACHE = {}


def load_split_features(split, feature_set):
    key = (split, feature_set)
    if key not in _CACHE:
        _CACHE[key] = np.load(os.path.join(CACHE_DIR, f"{split}_{feature_set.replace('+', '_')}.npy"))
    return _CACHE[key]


def trainval_features(feature_set):
    return np.vstack([load_split_features("train", feature_set), load_split_features("val", feature_set)])


def build_model(c, seed=42):
    if c["family"] == "tree":
        return build_tree_model(c["model_type"], c["params"], seed=seed)
    return build_sklearn_model(c["model_type"], c["params"], seed=seed)


def get_trainval_df_and_y():
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    return trainval, trainval["Solubility"].values.astype(np.float32)


def compute_oof_predictions(trainval, y):
    folds = scaffold_group_kfold(trainval, n_splits=N_FOLDS, seed=42)
    oof_preds = {c["name"]: np.zeros(len(y), dtype=np.float64) for c in CANDIDATES}
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"  OOF fold {fold_i}...")
        for c in CANDIDATES:
            X = trainval_features(c["feature_set"])
            model = build_model(c)
            model.fit(X[train_idx], y[train_idx])
            oof_preds[c["name"]][val_idx] = model.predict(X[val_idx])
    return oof_preds


def compute_our_blend_test_predictions():
    print("=" * 70)
    print("Part A: reproduce our weighted_blend ensemble's test predictions")
    print("=" * 70)
    trainval, y_trainval = get_trainval_df_and_y()
    oof_preds = compute_oof_predictions(trainval, y_trainval)
    weights = optimize_blend_weights(oof_preds, y_trainval)
    print(f"  blend weights (fit on train+val OOF only): {weights}")

    test_preds = {}
    for c in CANDIDATES:
        X_trainval = trainval_features(c["feature_set"])
        X_test = load_split_features("test", c["feature_set"])
        model = build_model(c)
        model.fit(X_trainval, y_trainval)
        test_preds[c["name"]] = model.predict(X_test)

    blend_test_preds = weighted_blend(test_preds, weights)
    y_test = np.load(os.path.join(CACHE_DIR, "test_y.npy"))
    m = regression_metrics(y_test, blend_test_preds)
    print(f"  weighted_blend TEST (sanity check, should match final_test_evaluation.py): {format_metrics(m)}")
    return blend_test_preds, y_test


def run_dissolvr_clean(y_true_from_cache):
    print("\n" + "=" * 70)
    print("Part B: clean rerun of the LITERAL dissolvr/ pipeline, 5 seeds")
    print("=" * 70)
    train_df = pd.read_csv(os.path.join(DISSOLVR_DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DISSOLVR_DATA_DIR, "test.csv"))

    featurizer = MoleculeFeaturizer()
    print("Featurizing train...")
    X_train = featurizer.transform(train_df["SMILES"])
    y_train = train_df["LogS"].values
    print("Featurizing test...")
    X_test = featurizer.transform(test_df["SMILES"])
    y_test = test_df["LogS"].values

    # Sanity check: dissolvr/data/test.csv's LogS should equal the cached
    # test_y.npy used everywhere else in this round, same row order.
    assert np.allclose(y_test, y_true_from_cache, atol=1e-6), \
        "dissolvr test labels don't match solubility_models' cached test_y.npy -- row order mismatch?"
    print("  Confirmed: dissolvr/data/test.csv labels match solubility_models test_y.npy (same order).")

    seed_preds = {}
    rmses, maes, r2s = [], [], []
    for seed in DISSOLVR_SEEDS:
        t0 = time.time()
        model = CatBoostRegressor(
            iterations=10000, learning_rate=0.02, depth=8, l2_leaf_reg=5,
            verbose=0, random_state=seed, allow_writing_files=False, thread_count=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        seed_preds[seed] = preds
        m = regression_metrics(y_test, preds)
        rmses.append(m["rmse"]); maes.append(m["mae"]); r2s.append(m["r2"])
        print(f"  seed {seed}: {format_metrics(m)}  ({time.time() - t0:.0f}s)")

    print(f"\n  DISSOLVR clean 5-seed: RMSE={np.mean(rmses):.4f} +/- {np.std(rmses):.4f}, "
          f"MAE={np.mean(maes):.4f} +/- {np.std(maes):.4f}, R2={np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")

    dissolvr_mean_preds = np.mean(np.column_stack(list(seed_preds.values())), axis=1)
    return dissolvr_mean_preds, y_test, dict(rmses=rmses, maes=maes, r2s=r2s)


def paired_bootstrap(y_true, preds_a, preds_b, name_a, name_b, n_boot=N_BOOTSTRAP, seed=42):
    """Resamples test rows (with replacement) jointly for both models,
    computing RMSE_a - RMSE_b each time. If the resulting distribution's 95%
    CI excludes 0, the RMSE gap is unlikely to be finite-test-set noise."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    rmse_a_samples = np.empty(n_boot)
    rmse_b_samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        rmse_a = np.sqrt(np.mean((yt - preds_a[idx]) ** 2))
        rmse_b = np.sqrt(np.mean((yt - preds_b[idx]) ** 2))
        rmse_a_samples[i] = rmse_a
        rmse_b_samples[i] = rmse_b
        diffs[i] = rmse_a - rmse_b

    print(f"\n{'=' * 70}")
    print(f"Part C: paired bootstrap ({n_boot} resamples) -- RMSE({name_a}) - RMSE({name_b})")
    print(f"{'=' * 70}")
    print(f"  {name_a} RMSE: point={np.sqrt(np.mean((y_true - preds_a) ** 2)):.4f}, "
          f"bootstrap 95% CI=[{np.percentile(rmse_a_samples, 2.5):.4f}, {np.percentile(rmse_a_samples, 97.5):.4f}]")
    print(f"  {name_b} RMSE: point={np.sqrt(np.mean((y_true - preds_b) ** 2)):.4f}, "
          f"bootstrap 95% CI=[{np.percentile(rmse_b_samples, 2.5):.4f}, {np.percentile(rmse_b_samples, 97.5):.4f}]")
    diff_point = np.sqrt(np.mean((y_true - preds_a) ** 2)) - np.sqrt(np.mean((y_true - preds_b) ** 2))
    ci_lo, ci_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    frac_b_better = float(np.mean(diffs > 0))
    print(f"  Difference (RMSE_{name_a} - RMSE_{name_b}): point={diff_point:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Fraction of {n_boot} resamples where {name_b} beats {name_a}: {frac_b_better:.4f}")
    if ci_lo > 0:
        print(f"  --> CI excludes 0: {name_b}'s RMSE advantage over {name_a} is statistically robust to test-set resampling.")
    else:
        print(f"  --> CI includes 0: cannot rule out that the observed gap is within test-set sampling noise.")
    return dict(diff_point=diff_point, ci_lo=ci_lo, ci_hi=ci_hi, frac_b_better=frac_b_better)


def main():
    blend_preds, y_test = compute_our_blend_test_predictions()
    dissolvr_preds, y_test_dissolvr, dissolvr_seed_stats = run_dissolvr_clean(y_test)

    result = paired_bootstrap(y_test, dissolvr_preds, blend_preds, "dissolvr_clean_5seed_mean", "our_weighted_blend")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"DISSOLVR clean rerun (literal dissolvr/ pipeline, 5 seeds): "
          f"RMSE={np.mean(dissolvr_seed_stats['rmses']):.4f} +/- {np.std(dissolvr_seed_stats['rmses']):.4f}")
    print(f"Our weighted_blend ensemble (single deterministic run): "
          f"RMSE={np.sqrt(np.mean((y_test - blend_preds) ** 2)):.4f}")
    print(f"Paired bootstrap 95% CI on the RMSE gap: [{result['ci_lo']:.4f}, {result['ci_hi']:.4f}] "
          f"({'excludes' if result['ci_lo'] > 0 else 'includes'} 0)")


if __name__ == "__main__":
    main()
