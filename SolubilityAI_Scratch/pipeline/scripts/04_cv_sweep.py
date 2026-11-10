"""Step 04: cross-validated sweep of tabular model families.  Usage: python 04_cv_sweep.py <stage>

Every configuration listed as a 'final' config is CV'd, refit on all train and SAVED immediately.
Hyperparameter-search trials (stage 'hp_*') are CV'd and logged to results/hpsearch/<family>.csv;
only the winning trial of each search is refit and saved (see REPORT for why).
All seeds fixed: model seed 42, HP-sampling seed 123.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner

stage = sys.argv[1]
HP_SEED = 123

SUFFIX = "" if runner.DATASET == "v1" else "_" + runner.DATASET   # dataset v2 -> "<name>_v2"

def go(name, fam, fs, p, note=""):
    name = name + SUFFIX
    res = R / "results/cv_results.csv"
    if res.exists() and name in set(pd.read_csv(res).name) and (R / "saved_models" / name / "config.json").exists():
        print("skip (already done):", name); return
    row, _ = runner.run(name, fam, fs, p, note=note)
    runner.log_result(row); print(json.dumps(row), flush=True)

def hp_search(tag, fam, fs, sampler, n_trials, final_name):
    rng = np.random.RandomState(HP_SEED)
    tag = tag + SUFFIX
    path = R / f"results/hpsearch/{tag}.csv"; path.parent.mkdir(exist_ok=True, parents=True)
    rows = []
    for t in range(n_trials):
        p = sampler(rng)
        row, _ = runner.run(f"{tag}_trial{t:02d}", fam, fs, p, save_model=False)
        rows.append(row); pd.DataFrame(rows).to_csv(path, index=False)
        print(t, row["cv_rmse_pooled"], p, flush=True)
    best = min(rows, key=lambda r: r["cv_rmse_pooled"])
    go(final_name, fam, fs, json.loads(best["params"]), note=f"best of {n_trials}-trial random search {path.name}")

if stage == "baselines":
    go("esol4_linear", "linear", "esol4", {}, "Delaney-style 4-descriptor linear baseline")
    go("ridge_desc", "ridge", "rdkit_desc", {"alpha": 10.0})
    go("knn_morganbits_k5", "knn", "morgan_bits_r2_2048", {"k": 5}, "Tanimoto (Jaccard) kNN")
    go("rf_morgan", "rf", "morgan_counts_r2_2048", {"n_estimators": 500, "max_features": 0.3, "min_samples_leaf": 1})
    go("rf_desc", "rf", "rdkit_desc", {"n_estimators": 500, "max_features": 0.3, "min_samples_leaf": 1})
    go("rf_desc_morgan", "rf", "desc+morgan_counts_r2_2048", {"n_estimators": 500, "max_features": 0.2, "min_samples_leaf": 1})

elif stage == "hp_svr":
    hp_search("svr_desc", "svr", "rdkit_desc",
              lambda r: {"C": float(10 ** r.uniform(0, 2)), "gamma": float(10 ** r.uniform(-3.5, -2)),
                         "epsilon": float(r.choice([0.05, 0.1, 0.2]))}, 10, "svr_desc")
elif stage == "hp_mlp":
    hp_search("mlp_desc", "mlp", "rdkit_desc",
              lambda r: {"hidden": [int(r.choice([256, 512])), int(r.choice([128, 256]))],
                         "alpha": float(10 ** r.uniform(-3, 0)), "lr": 1e-3, "max_iter": 200}, 6, "mlp_desc")
elif stage == "hp_lgbm_desc":
    hp_search("lgbm_desc", "lgbm", "rdkit_desc",
              lambda r: {"n_estimators": int(r.choice([800, 1500, 2500])), "learning_rate": float(r.choice([0.01, 0.02, 0.04])),
                         "num_leaves": int(r.choice([15, 31, 63])), "min_child_samples": int(r.choice([5, 10, 20, 40])),
                         "subsample": float(r.uniform(0.6, 1.0)), "subsample_freq": 1,
                         "colsample_bytree": float(r.uniform(0.2, 0.7)), "reg_lambda": float(10 ** r.uniform(-2, 1))}, 14, "lgbm_desc")
elif stage == "hp_lgbm_dm":
    hp_search("lgbm_desc_morgan", "lgbm", "desc+morgan_counts_r2_2048",
              lambda r: {"n_estimators": int(r.choice([1500, 2500])), "learning_rate": float(r.choice([0.02, 0.04])),
                         "num_leaves": int(r.choice([15, 31, 63])), "min_child_samples": int(r.choice([5, 10, 20])),
                         "subsample": float(r.uniform(0.6, 1.0)), "subsample_freq": 1,
                         "colsample_bytree": float(r.uniform(0.1, 0.4)), "reg_lambda": float(10 ** r.uniform(-2, 1))}, 8, "lgbm_desc_morgan")
elif stage == "hp_xgb_dm":
    hp_search("xgb_desc_morgan", "xgb", "desc+morgan_counts_r2_2048",
              lambda r: {"n_estimators": int(r.choice([1000, 2000])), "learning_rate": float(r.choice([0.02, 0.04])),
                         "max_depth": int(r.choice([5, 7, 9])), "min_child_weight": float(r.choice([1, 3, 6])),
                         "subsample": float(r.uniform(0.6, 1.0)), "colsample_bytree": float(r.uniform(0.1, 0.4)),
                         "reg_lambda": float(10 ** r.uniform(-1, 1))}, 6, "xgb_desc_morgan")
elif stage == "catboost":
    go("catboost_desc", "catboost", "rdkit_desc", {"iterations": 3000, "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 3},
       "single literature-default-ish config (no search: CatBoost CV is slow on 2 CPUs)")
elif stage == "v2_fixed":
    # v2 re-runs of the strongest v1 families, reusing the v1-selected hyperparameters (no new search)
    P = pd.read_csv(R / "results/cv_results.csv").set_index("name").params
    for n, fam, fs in [("lgbm_desc", "lgbm", "rdkit_desc"), ("lgbm_desc_morgan", "lgbm", "desc+morgan_counts_r2_2048"),
                       ("xgb_desc_morgan", "xgb", "desc+morgan_counts_r2_2048"), ("svr_desc", "svr", "rdkit_desc"),
                       ("rf_desc", "rf", "rdkit_desc"), ("catboost_desc", "catboost", "rdkit_desc")]:
        go(n, fam, fs, json.loads(P[n]), note="v2 domain-filtered training set; hyperparameters from v1 search")
elif stage == "mordred":
    P = pd.read_csv(R / "results/cv_results.csv").set_index("name").params
    go("lgbm_mordred", "lgbm", "mordred", json.loads(P["lgbm_desc"]), "lgbm_desc hyperparameters on Mordred 2D descriptors")
    go("lgbm_desc_mordred", "lgbm", "desc+mordred", json.loads(P["lgbm_desc"]), "lgbm_desc hyperparameters on RDKit + Mordred descriptors")
elif stage == "extra":
    go("et_desc_morgan", "et", "desc+morgan_counts_r2_2048", {"n_estimators": 500, "max_features": 0.3, "min_samples_leaf": 1})
    go("lgbm_desc_maccs", "lgbm", "desc+maccs", json.loads(pd.read_csv(R / "results/cv_results.csv").set_index("name").loc["lgbm_desc" + SUFFIX, "params"]),
       "lgbm_desc hyperparameters, + MACCS keys")
