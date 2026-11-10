"""Step 09: write per-model README.md + performance.json and the saved_models/README.md index
from recorded artifacts (config.json, results/07_test_metrics.csv, results/08_verification.json,
results/07_external_sc.json).  The recommendation paragraph is read from
results/09_recommendation.md (written by hand after inspecting results)."""
import json
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parents[1]
SM = R / "saved_models"
def _both(f1, f2):   # round-1 rows + round-2 rows (round-2 file also repeats blend_nnls as reference)
    a = pd.read_csv(R / f1)
    if (R / f2).exists():
        b = pd.read_csv(R / f2); a = pd.concat([a, b[~b.name.isin(a.name)]])
    return a.set_index("name")
tm = _both("results/07_test_metrics.csv", "results/r2_07_test_metrics.csv")          # in-domain test (primary)
tf = _both("results/07_test_metrics_fulltest.csv", "results/r2_07_test_metrics_fulltest.csv")  # full curated test
if (R / "results/r2_07b_test_metrics.json").exists():   # round-2 bug-fix replacements
    _b = pd.read_json(R / "results/r2_07b_test_metrics.json")
    tm = pd.concat([tm, _b[_b.test_set == "indomain"].drop(columns="test_set").set_index("name")])
    tf = pd.concat([tf, _b[_b.test_set == "full"].drop(columns="test_set").set_index("name")])
cvd = (pd.read_csv(R / "results/r2_07_cv_indomain_all.csv") if (R / "results/r2_07_cv_indomain_all.csv").exists()
       else pd.read_csv(R / "results/07_cv_indomain.csv")).set_index("name")
_dom = pd.read_csv(R / "data/06b_train_v2.csv")[["mol_id"]]
for _f in sorted(SM.iterdir()):     # configs added after the CV summary files (round-2 bug fix): compute in-domain CV from OOF
    if (_f / "config.json").exists() and _f.name not in cvd.index:
        _o = _dom.merge(pd.read_csv(R / f"results/oof/{_f.name}.csv"), on="mol_id")
        _e = _o.y - _o.oof
        cvd.loc[_f.name, "cv_rmse_indomain"] = float((_e ** 2).mean() ** 0.5)
        cvd.loc[_f.name, "cv_r2_indomain"] = float(1 - (_e ** 2).sum() / ((_o.y - _o.y.mean()) ** 2).sum())
ver = json.load(open(R / "results/08_verification.json"))["C_model_reload"]
for _f in ["results/r2_08_verification.json", "results/r2_08b_verification.json"]:
    if (R / _f).exists(): ver.update(json.load(open(R / _f))["C_model_reload"])
R2EXT = json.load(open(R / "results/r2_07_external_sc.json")) if (R / "results/r2_07_external_sc.json").exists() else {}
ext = json.load(open(R / "results/07_external_sc.json"))
pre = json.load(open(R / "results/07_preregistration.json"))
f3 = lambda x: f"{x:.3f}"

rows = []
for folder in sorted(SM.iterdir()):
    if not (folder / "config.json").exists(): continue
    c = json.load(open(folder / "config.json")); n = c["name"]
    t = tm.loc[n] if n in tm.index else None
    v = ver.get(n, {"status": "NOT VERIFIED"})
    perf = dict(cv=c["cv"], test=None if t is None else dict(
        rmse=t.test_rmse, r2=t.test_r2, mae=t.test_mae, rmse_ci95_molecule_bootstrap=eval(t.rmse_ci_mol) if isinstance(t.rmse_ci_mol, str) else t.rmse_ci_mol,
        r2_ci95_molecule_bootstrap=eval(t.r2_ci_mol) if isinstance(t.r2_ci_mol, str) else t.r2_ci_mol,
        rmse_ci95_cluster_bootstrap=eval(t.rmse_ci_cluster) if isinstance(t.rmse_ci_cluster, str) else t.rmse_ci_cluster),
        external={k: (ext[k].get(n) or R2EXT.get(k, {}).get(n)) for k in ext}, reload_verification=v)
    json.dump(perf, open(folder / "performance.json", "w"), indent=2, default=float)
    files = sorted(p.name for p in folder.iterdir())
    fam = c["family"]
    if fam == "stack":
        what = (f"Stacked ensemble: a {c['stacker']} meta-model over the predictions of " + ", ".join(f"`{x}`" for x in c["components"]) +
                (" plus meta features " + ", ".join(c["meta_desc"]) + " and nn_sim" if c.get("meta_desc") else "") +
                ".  Loading it loads each component folder (siblings in saved_models/); logic in stack_predict.py.")
        feat = "Each component featurizes the SMILES with its own featurize.py; meta features use this folder's featurize.py."
        pre_txt = "Stacker file (stacker.joblib or stacker.lgbm.txt) in this folder; components carry their own preprocessors."
    elif fam == "blend":
        what = "Weighted blend of saved models: " + ", ".join(f"`{x['folder']}` x {x['weight']:.3f}" for x in c["components"]) + \
               f" + intercept {c['intercept']:.4f}.  Loading this model loads each component folder (siblings in saved_models/)."
        feat = "Each component featurizes the SMILES with its own featurize.py."
        pre_txt = "None of its own (each component carries its own preprocessor)."
    else:
        what = f"{fam} model on feature set `{c['feature_set']}`."
        feat = f"`featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then " + \
               ("`graphs()` (atom/bond features, see ATOM_LIST/atom_features/bond_features)" if fam.startswith("gnn") else f"`featurize(smiles, '{c['feature_set']}', desc_names)`") + \
               (" + descriptors in the exact order of `config.json:desc_names`" if c.get("desc_names") else "") + "."
        pre_txt = "`preprocessor.joblib` (sklearn Pipeline, fitted on the full training set)" if (folder / "preprocessor.joblib").exists() \
            else "None required (model handles raw features / NaN natively)."
    hp = c.get("params", c.get("arch"))
    readme = f"""# {n}

**What it is:** {what}

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/{n}/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/{n}")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** {feat}

**Preprocessor:** {pre_txt}

**Hyperparameters / config:** see `config.json` (`{json.dumps(hp)[:400]}`); seed(s) {c.get('seeds', c.get('seed'))}.
Trained on: {c.get('trained_on', 'blend of the component models (weights fitted on in-domain train OOF predictions)')}; script `{c.get('training_script')}`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | {f3(c['cv']['pooled']['rmse'])} |
| CV RMSE / R², in-domain train molecules (comparable across models) | {f3(cvd.cv_rmse_indomain[n])} / {f3(cvd.cv_r2_indomain[n])} |
| In-domain test RMSE [95% CI, cluster bootstrap] (n={int(tm.n_test.iloc[0])}) | {'n/a' if t is None else f3(t.test_rmse) + ' ' + str([round(x,3) for x in perf['test']['rmse_ci95_cluster_bootstrap']])} |
| In-domain test R² [95% CI, molecule bootstrap] | {'n/a' if t is None else f3(t.test_r2) + ' ' + str([round(x,3) for x in perf['test']['r2_ci95_molecule_bootstrap']])} |
| Full curated test RMSE / R² (n={int(tf.n_test.iloc[0])}, incl. metal/mixture entries) | {f3(tf.test_rmse[n])} / {f3(tf.test_r2[n])} |
| SC2-tight RMSE / R² (logS0, see caveats) | {'n/a' if not perf['external'].get('SC2_tight_processed') else f3(perf['external']['SC2_tight_processed']['rmse']) + ' / ' + f3(perf['external']['SC2_tight_processed']['r2'])} |

**Reload verification (fresh process):** {v.get('status')} — reference max |Δ| = {v.get('ref_max_abs_diff', 'n/a')}, test RMSE reproduced = {v.get('test_rmse', 'n/a')}.

**Files:** {', '.join(files)}

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
"""
    open(folder / "README.md", "w").write(readme)
    rows.append(dict(name=n, family=fam, feature_set=c.get("feature_set", "(components)"),
                     cv_rmse=cvd.cv_rmse_indomain[n], cv_r2=cvd.cv_r2_indomain[n], dataset=c.get("dataset", "v1" if fam not in ("blend", "stack") else "v1+v2 components"),
                     full_rmse=tf.test_rmse[n],
                     test_rmse=None if t is None else t.test_rmse, test_r2=None if t is None else t.test_r2,
                     verified=v.get("status")))
idx = pd.DataFrame(rows).sort_values("cv_rmse")
tab = "| model (folder) | family | features | train set | CV RMSE (in-domain) | CV R² | Test RMSE (in-domain) | Test R² | Full-test RMSE | reload check |\n|---|---|---|---|---|---|---|---|---|---|\n"
for r in idx.itertuples():
    tab += f"| [`{r.name}`]({r.name}/) | {r.family} | {r.feature_set} | {r.dataset} | {r.cv_rmse:.3f} | {r.cv_r2:.3f} | " \
           f"{r.test_rmse:.3f} | {r.test_r2:.3f} | {r.full_rmse:.3f} | {r.verified} |\n"
rec = open(R / "results/09_recommendation.md").read() if (R / "results/09_recommendation.md").exists() else "(pending)"
open(SM / "README.md", "w").write(f"""# Saved models — index

Every model configuration trained and evaluated in this pipeline, sorted by cross-validated RMSE
on the in-domain training molecules (5-fold grouped CV; the same molecules and folds for every
model, so the column is comparable; lower is better).  Train set v1 = full curated AqSolDB
(minus SC overlap); v2 = v1 minus metal-containing and multi-component-mixture entries.
Test columns: in-domain test (n=1804, primary) and full curated test (n=1852).  Test metrics are from the single,
pre-registered held-out evaluation of each round (round 1: `scripts/07_test_eval.py`, pick `{pre['selected']}`;
round 2 = names starting `r2_` plus `gnn_*_s43/s44`: `scripts/r2_07_test_eval.py`, pick `r2_blend_nnls`).
Each configuration was predicted on the test set exactly once.
"reload check" = fresh-process reload + reproduction of reference predictions and test metrics
(`scripts/08_verify.py`, `scripts/r2_08_verify.py`).  Units: LogS, log10(mol/L).

{tab}
## Recommendation for UI integration

{rec}

## Not persisted (disclosed)

* Per-trial models from the random hyperparameter searches (`results/hpsearch/*.csv` lists every
  trial's parameters and CV score); only the winning configuration of each search was refit on
  all training data and saved.  Any trial can be re-created exactly from its logged parameters
  and seed with `src/runner.py`.
* Per-fold CV models (5 per configuration): used only to produce out-of-fold predictions
  (`results/oof/*.csv`), then discarded.  No saved model is unrecoverable.
* `gnn_hybrid_v2` was planned but cancelled at start, before any fold finished (see
  `logs/DECISIONS.log`): it was never trained or evaluated, so there is nothing to save.
* Round 2: `gnn_hybrid_ens3` (one 3-seed run) and `r2_lgbm_desc_mordred_avalon_erg_huber` were
  cancelled mid-CV, before producing any result (see `logs/DECISIONS.log`).  The 3-seed ensemble was
  built instead from single-seed runs (`r2_gnn_hybrid_ens3`).  Four CV-only exploratory stackers (residual
  correction and gated blends; `scripts/r2_06b_*`, `scripts/r2_06c_*`) were scored in nested CV, did not
  beat the plain NNLS blend, and were not refit or saved; their scores are in `results/r2_log.csv`.

## Every folder contains
`config.json` (family, feature set, descriptor order, hyperparameters, seeds, library versions, CV
metrics), the native model file, `preprocessor.joblib` where applicable, `featurize.py` (exact
featurization code), `predict.py` (loader + CLI), `reference_predictions.csv`, `performance.json`,
and a README.
""")
print(idx.to_string())
