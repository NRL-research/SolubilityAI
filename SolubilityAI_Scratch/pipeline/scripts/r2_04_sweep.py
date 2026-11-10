"""Round 2, step 04: new feature representations / model configurations.
Same protocol as round 1: 5-fold grouped CV on the TRAIN split only (same folds), refit on all
train, saved immediately via src/runner.py.  Trained on dataset v1 (round 1 showed v1 and v2
training are equivalent in-domain and v1 is better out of domain).  The test set is not read.
Usage: python r2_04_sweep.py <stage>
Every configuration and its in-domain CV score is appended to results/r2_log.csv with a note
on the motivation.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner

stage = sys.argv[1]
P = pd.read_csv(R / "results/cv_results.csv").set_index("name").params
LGB = json.loads(P["lgbm_desc"])          # round-1 tuned LightGBM settings, reused for comparability
DOM = pd.read_csv(R / "data/06b_train_v2.csv")[["mol_id"]]

def indomain(name):
    o = DOM.merge(pd.read_csv(R / f"results/oof/{name}.csv"), on="mol_id")
    return runner.metrics(o.y, o.oof)

def go(name, fam, fs, p, why):
    res = R / "results/cv_results.csv"
    if name in set(pd.read_csv(res).name) and (R / "saved_models" / name / "config.json").exists():
        print("skip", name); return
    row, _ = runner.run(name, fam, fs, p, note=why, hp_search_log="scripts/r2_04_sweep.py " + stage)
    runner.log_result(row)
    m = indomain(name)
    log = R / "results/r2_log.csv"
    rec = pd.DataFrame([dict(name=name, family=fam, feature_set=fs, params=json.dumps(p), why=why,
                             cv_rmse_indomain=m["rmse"], cv_r2_indomain=m["r2"], cv_mae_indomain=m["mae"],
                             cv_rmse_fold_sd_own=row["cv_rmse_fold_sd"], secs=row["secs"])])
    if log.exists():
        old = pd.read_csv(log); rec = pd.concat([old[old.name != name], rec])
    rec.to_csv(log, index=False)
    print(f"{name}: in-domain CV RMSE {m['rmse']:.4f}  R2 {m['r2']:.4f}  ({row['secs']:.0f}s)", flush=True)

if stage == "features":
    go("r2_lgbm_dm_huber", "lgbm", "desc+mordred", dict(LGB, objective="huber", alpha=1.0),
       "Error analysis: many top errors are label/structure errors (e.g. propyzamide given as propanamide). "
       "A Huber loss caps the influence of large residuals during training.")
    go("r2_lgbm_d_crip_pack", "lgbm", "r2:desc+crippen_types+packing", LGB,
       "Group-contribution atom types + crystal-packing/melting-point proxies (MP is the missing term of the "
       "General Solubility Equation; errors grow with aromatic ring count).")
    go("r2_lgbm_dm_crip_pack", "lgbm", "r2:desc+mordred+crippen_types+packing", LGB,
       "Same additions on top of the best round-1 feature set.")
    go("r2_lgbm_m2v", "lgbm", "r2:mol2vec", LGB, "Pretrained chemical embedding (mol2vec) alone.")
    go("r2_lgbm_dm_m2v", "lgbm", "r2:desc+mordred+mol2vec", LGB, "Pretrained embedding added to the best feature set.")
    go("r2_lgbm_d_ap_tt", "lgbm", "r2:desc+atompair+torsion", LGB,
       "Alternative fingerprints (atom-pair, topological torsion) that encode longer-range topology than Morgan.")
    go("r2_lgbm_d_av_erg", "lgbm", "r2:desc+avalon+erg", LGB, "Avalon substructure counts + ErG pharmacophore graph.")
elif stage == "models":
    go("r2_catboost_dm", "catboost", "desc+mordred", {"iterations": 3000, "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 3},
       "CatBoost on the best feature set (round 1 only ran CatBoost on RDKit descriptors).")
    go("r2_svr_m2v", "svr", "r2:mol2vec", {"C": 10.0, "gamma": float(1 / 301), "epsilon": 0.05},
       "Kernel model on the pretrained embedding: a different inductive bias for blending.")
    go("r2_svr_dm", "svr", "desc+mordred", {"C": 10.0, "gamma": float(1 / 1526), "epsilon": 0.05},
       "SVR on the larger descriptor set (gamma = 1/n_features, the same scaling as round-1 SVR).")
elif stage == "ergfix":
    # bug fix: ErG ('erg') was non-deterministic (see featurize.erg2); retrain the two ErG models with 'erg2'
    go("r2_lgbm_d_av_erg2", "lgbm", "r2:desc+avalon+erg2", LGB, "Bug-fix retrain of r2_lgbm_d_av_erg with deterministic ErG (erg2).")
    go("r2_lgbm_desc_mordred_avalon_erg2", "lgbm", "r2:desc+mordred+avalon+erg2", LGB,
       "Bug-fix retrain of r2_lgbm_desc_mordred_avalon_erg with deterministic ErG (erg2).")
elif stage == "combo":
    # chosen from the 'features' stage by in-domain CV only (see results/r2_log.csv)
    blocks = sys.argv[2]
    go("r2_lgbm_" + blocks.replace("+", "_"), "lgbm", "r2:" + blocks, LGB, "Combination of blocks that helped in CV.")
    go("r2_lgbm_" + blocks.replace("+", "_") + "_huber", "lgbm", "r2:" + blocks, dict(LGB, objective="huber", alpha=1.0),
       "Same combination with the Huber loss.")
