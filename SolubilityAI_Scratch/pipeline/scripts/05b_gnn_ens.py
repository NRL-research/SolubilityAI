"""Step 05b: SEED-ENSEMBLE variant of step 05 (identical pipeline; each fit = mean of several seeds;
also exposes dropout / weight decay).  Usage: python 05b_gnn_ens.py <name> <hybrid> <epochs> <seeds e.g. 42,43,44> <dropout> <wd>
--- original step-05 docstring ---
Step 05: message-passing GNN (graph-only and graph+descriptor hybrid), 5-fold grouped CV on
train, refit on all train, save immediately.  Fixed epoch budget (no early stopping, so the
validation fold is never used for training decisions).
Usage: python 05_gnn.py <name> <hybrid 0|1> [epochs] [seed]
"""
import json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
import featurize as F, gnn_model as G, runner

name, hybrid = sys.argv[1], bool(int(sys.argv[2]))
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
SEEDS = [int(x) for x in sys.argv[4].split(",")]
DROPOUT, WD = float(sys.argv[5]), float(sys.argv[6])
SEED = SEEDS[0]
torch.set_num_threads(2)
LOG = open(R / f"logs/05_{name}.log", "a")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

df = runner.data()["df"]
tr = np.where(df.split == "train")[0]
y = df.LogS.values[tr]; folds = df.cv_fold.values[tr]
graphs = F.graphs(df.SMILES.values[tr].tolist())
D = runner.features("rdkit_desc", tr) if hybrid else None
arch = dict(n_atom=F.N_ATOM_FEAT, n_bond=F.N_BOND_FEAT, hidden=200, layers=4,
            n_global=(D.shape[1] if hybrid else 0), dropout=DROPOUT, head=256)
hp = dict(epochs=EPOCHS, lr=1e-3, bs=64, wd=WD)
def make_pre():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("qt", QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0))])

def fit_predict(fi, va_idx=None):
    ym, ys = y[fi].mean(), y[fi].std()
    pre = gx = None
    if hybrid:
        pre = make_pre().fit(D[fi]); gx = pre.transform(D).astype(np.float32)
    gtr = [graphs[i] for i in fi]
    nets = [G.train(gtr, (y[fi] - ym) / ys, None if gx is None else gx[fi], seed=s, arch=arch, log=log, **hp) for s in SEEDS]
    p = None
    if va_idx is not None:
        p = np.mean([G.predict_graphs(n, [graphs[i] for i in va_idx], None if gx is None else gx[va_idx]) for n in nets], 0) * ys + ym
    return nets, pre, ym, ys, p

t0 = time.time()
oof = np.full(len(tr), np.nan); per = []
for k in range(5):
    fi, va = np.where(folds != k)[0], np.where(folds == k)[0]
    log(f"--- fold {k}")
    _, _, _, _, p = fit_predict(fi, va)
    oof[va] = p; per.append(runner.metrics(y[va], p)); log(f"fold {k} {per[-1]}")
pooled = runner.metrics(y, oof); log(f"pooled {pooled}")
pd.DataFrame(dict(mol_id=df.mol_id.values[tr], cv_fold=folds, y=y, oof=oof)).to_csv(R / f"results/oof/{name}.csv", index=False)

# ---- final refit + save ----
log("--- final fit on all train")
nets, pre, ym, ys, _ = fit_predict(np.arange(len(tr)))
folder = R / "saved_models" / name
if folder.exists(): shutil.rmtree(folder)
folder.mkdir(parents=True)
files = []
for s_, n_ in zip(SEEDS, nets):
    torch.save(n_.state_dict(), folder / f"model_seed{s_}.pt"); files.append(f"model_seed{s_}.pt")
if hybrid:
    joblib.dump(pre, folder / "preprocessor.joblib"); files.append("preprocessor.joblib")
for f in ["featurize.py", "gnn_model.py"]: shutil.copy(R / "src" / f, folder / f)
shutil.copy(R / "src/predict_template.py", folder / "predict.py")
fam = "gnn_hybrid" if hybrid else "gnn"
cfg = dict(name=name, family=fam, feature_set="graph" + ("+rdkit_desc(quantile-normal)" if hybrid else ""),
           arch=arch, train_hparams=hp, seeds=SEEDS, y_mean=float(ym), y_std=float(ys),
           desc_names=[str(s) for s in runner.data()["z"]["desc_names"]] if hybrid else None,
           input="SMILES string(s)", output="predicted LogS (log10 mol/L)",
           trained_on=f"{runner.TRAIN_FILE} (n={len(tr)})", dataset=runner.DATASET, cv=dict(per_fold=per, pooled=pooled),
           model_files=files, versions=dict(runner.versions(), torch=torch.__version__),
           training_script="scripts/05b_gnn_ens.py " + " ".join(sys.argv[1:]))
json.dump(cfg, open(folder / "config.json", "w"), indent=2)
ref = runner.reference_smiles()
gx = pre.transform(F.featurize(ref, "rdkit_desc", cfg["desc_names"])).astype(np.float32) if hybrid else None
rp = np.mean([G.predict_graphs(n_, F.graphs(ref), gx) for n_ in nets], 0) * ys + ym
pd.DataFrame(dict(smiles=ref, pred_at_save=rp)).to_csv(folder / "reference_predictions.csv", index=False)
row = dict(name=name, family=fam, feature_set=cfg["feature_set"], params=json.dumps(dict(arch=arch, seeds=SEEDS, **hp)), seed=SEED,
           cv_rmse_pooled=pooled["rmse"], cv_r2_pooled=pooled["r2"], cv_mae_pooled=pooled["mae"],
           cv_rmse_fold_mean=float(np.mean([p["rmse"] for p in per])), cv_rmse_fold_sd=float(np.std([p["rmse"] for p in per], ddof=1)),
           cv_r2_fold_mean=float(np.mean([p["r2"] for p in per])), cv_r2_fold_sd=float(np.std([p["r2"] for p in per], ddof=1)),
           note=f"fixed-epoch MPNN, {len(SEEDS)}-seed ensemble", secs=time.time() - t0)
runner.log_result(row); log(json.dumps(row))
