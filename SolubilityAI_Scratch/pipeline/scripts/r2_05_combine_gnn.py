"""Round 2, step 05: build a seed-ensemble GNN from independently trained single-seed runs.

  python r2_05_combine_gnn.py <new_name> <member1> <member2> ...
Members must share architecture, training data, target scaling and preprocessor (checked).
The ensemble prediction (and its OOF prediction) is the mean of the members' predictions in the
standardized target scale, which is exactly what a multi-seed run of scripts/05b_gnn_ens.py computes.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import runner

name, members = sys.argv[1], sys.argv[2:]
cfgs = [json.load(open(R / f"saved_models/{m}/config.json")) for m in members]
for c in cfgs[1:]:
    for k in ["arch", "train_hparams", "y_mean", "y_std", "desc_names", "family", "trained_on"]:
        assert c[k] == cfgs[0][k], (k, c["name"])
hybrid = cfgs[0]["family"] == "gnn_hybrid"
if hybrid:   # preprocessor must be identical (deterministic fit on identical data)
    X = runner.features("rdkit_desc", np.arange(200))
    T = [joblib.load(R / f"saved_models/{m}/preprocessor.joblib").transform(X) for m in members]
    assert all(np.allclose(T[0], t) for t in T[1:])
folder = R / "saved_models" / name
if folder.exists(): shutil.rmtree(folder)
folder.mkdir(parents=True)
seeds, files = [], []
for m, c in zip(members, cfgs):
    for s in c["seeds"]:
        shutil.copy(R / f"saved_models/{m}/model_seed{s}.pt", folder / f"model_seed{s}.pt"); seeds.append(s); files.append(f"model_seed{s}.pt")
for f in ["featurize.py", "gnn_model.py", "predict.py"] + (["preprocessor.joblib"] if hybrid else []):
    shutil.copy(R / f"saved_models/{members[0]}/{f}", folder / f)
if hybrid: files.append("preprocessor.joblib")
O = [pd.read_csv(R / f"results/oof/{m}.csv") for m in members]
for o in O[1:]: assert (o.mol_id.values == O[0].mol_id.values).all()
oof = np.mean([o.oof.values for o in O], axis=0); y = O[0].y.values; folds = O[0].cv_fold.values
per = [runner.metrics(y[folds == k], oof[folds == k]) for k in range(5)]; pooled = runner.metrics(y, oof)
cfg = dict(cfgs[0], name=name, seeds=seeds, model_files=files, cv=dict(per_fold=per, pooled=pooled),
           members=members, training_script="scripts/r2_05_combine_gnn.py " + " ".join(sys.argv[1:]),
           note="seed ensemble = mean of member networks' standardized predictions")
json.dump(cfg, open(folder / "config.json", "w"), indent=2)
ref = pd.read_csv(R / f"saved_models/{members[0]}/reference_predictions.csv")
refs = [pd.read_csv(R / f"saved_models/{m}/reference_predictions.csv") for m in members]
assert all((r.smiles == ref.smiles).all() for r in refs)
pd.DataFrame(dict(smiles=ref.smiles, pred_at_save=np.mean([r.pred_at_save.values for r in refs], axis=0))).to_csv(folder / "reference_predictions.csv", index=False)
pd.DataFrame(dict(mol_id=O[0].mol_id, cv_fold=folds, y=y, oof=oof)).to_csv(R / f"results/oof/{name}.csv", index=False)
row = dict(name=name, family=cfg["family"], feature_set=cfg["feature_set"], params=json.dumps(dict(arch=cfg["arch"], seeds=seeds, **cfg["train_hparams"])),
           seed=seeds[0], cv_rmse_pooled=pooled["rmse"], cv_r2_pooled=pooled["r2"], cv_mae_pooled=pooled["mae"],
           cv_rmse_fold_mean=float(np.mean([p["rmse"] for p in per])), cv_rmse_fold_sd=float(np.std([p["rmse"] for p in per], ddof=1)),
           cv_r2_fold_mean=float(np.mean([p["r2"] for p in per])), cv_r2_fold_sd=float(np.std([p["r2"] for p in per], ddof=1)),
           note=f"{len(seeds)}-seed ensemble of {members}")
runner.log_result(row)
dom = pd.read_csv(R / "data/06b_train_v2.csv")[["mol_id"]].merge(pd.DataFrame(dict(mol_id=O[0].mol_id, y=y, oof=oof)), on="mol_id")
m = runner.metrics(dom.y, dom.oof)
log = R / "results/r2_log.csv"; d = pd.read_csv(log)
d = pd.concat([d[d.name != name], pd.DataFrame([dict(name=name, family=cfg["family"], feature_set=cfg["feature_set"], params=f"seeds {seeds}",
     why="Seed averaging: GNNs carry large blend weights and single networks are noisy (one seed, fixed epochs).",
     cv_rmse_indomain=m["rmse"], cv_r2_indomain=m["r2"], cv_mae_indomain=m["mae"], cv_rmse_fold_sd_own=row["cv_rmse_fold_sd"], secs=0)])])
d.to_csv(log, index=False)
print(name, "in-domain CV", m)
