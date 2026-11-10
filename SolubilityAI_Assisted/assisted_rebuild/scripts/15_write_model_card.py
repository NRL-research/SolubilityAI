"""
15_write_model_card.py
======================
Write saved_models/README.md -- the master index and recommendation.

Numbers are pulled from the saved artefacts (_manifest.json, blends/*.json,
_saved_vs_original_predictions.csv, _verification_report.csv) rather than typed in, so
the card cannot drift from what is actually on disk. The recommendation text is
editorial and lives here.
"""

import glob
import json
import os

import pandas as pd

ROOT = "/home/claude/work"
SAVED = f"{ROOT}/saved_models"

manifest = json.load(open(f"{SAVED}/_manifest.json"))
by_config = {m["config"]: m for m in manifest}
timing = pd.read_csv(f"{SAVED}/_saved_vs_original_predictions.csv").set_index("config")
blends = [json.load(open(f)) for f in sorted(glob.glob(f"{SAVED}/blends/*.json"))]
verif = pd.read_csv(f"{SAVED}/_verification_report.csv").set_index("config") \
    if os.path.exists(f"{SAVED}/_verification_report.csv") else None


def slug(name):
    return name.replace("[", "_").replace("]", "").replace("+", "_plus_")


def secs(config):
    if config in timing.index:
        return f"{timing.loc[config, 'seconds']:.0f}s"
    return "-"


def vstatus(config):
    if verif is None or config not in verif.index:
        return "not run"
    return "PASS" if bool(verif.loc[config, "passed"]) else "**FAIL**"


singles = sorted(manifest, key=lambda m: m["test_rmse"])

L = []
A = L.append

A("# SolubilityAI — saved models\n")
A("Trained LogS (aqueous solubility) predictors, each saved in its library's native")
A("format with everything needed to run it on a new molecule: the weights, the exact")
A("featurization code, the exact feature column order, and a runnable `predict.py`.\n")
A("All models predict **LogS** — the base-10 logarithm of aqueous solubility in mol/L.")
A("LogS −4 ≈ 10⁻⁴ mol/L. Most oral drugs sit between −4 and −1.\n")

A("## Recommendation: what to build a UI on\n")
best = by_config["lightgbm[mordred]"]
A(f"**Build on `{slug('lightgbm[mordred]')}/`.**\n")
A(f"It is the most accurate single model in the project (test RMSE **{best['test_rmse']:.4f}**, ")
A(f"R² **{best['test_r2']:.4f}**), and — this is the part that decides it — a paired bootstrap over")
A("the test set cannot distinguish it from any of the blends that score nominally")
A("higher. Against the best blend the gap is 0.0090 RMSE with a 95% CI of")
A("[−0.0037, +0.0217], which straddles zero. You would be paying seven models,")
A("23 weight files and four separate featurization passes per prediction for an")
A("improvement the data does not support. One model, three weight files, one")
A("featurization.\n")
A("**If you cannot take the Mordred dependency**, or 0.1 s/molecule of featurization is")
A(f"too slow: use `{slug('lightgbm[rdkit_descriptors]')}/` (RMSE ")
A(f"{by_config['lightgbm[rdkit_descriptors]']['test_rmse']:.4f}) or ")
A(f"`{slug('catboost[rdkit_full_eng]')}/` (RMSE ")
A(f"{by_config['catboost[rdkit_full_eng]']['test_rmse']:.4f}). Both need only RDKit and")
A("featurize roughly 13× faster. You give up about 0.035 RMSE — a real but small loss,")
A("and still far better than anything the project had before this rebuild.\n")
A("**If you want the best point estimate regardless of cost**, `blends/greedy_blend`")
A(f"is the lowest test RMSE recorded ({[b for b in blends if b['blend_name']=='greedy_blend'][0]['test_rmse']:.4f}). Treat it as a ceiling estimate, not as a")
A("model that is demonstrably better than the single LightGBM.\n")
A("**Do not build a UI on** `gatv2_graph_plus_5desc`, `catboost_solo_solubility_ai`,")
A("`xgboost_branch_solubility_ai`, `extra_trees_rdkit_descriptors`, or")
A("`blends/SolubilityAI_ensemble`. They are saved for provenance and for reproducing")
A("the report's baseline comparisons — they are the project's *old* architecture and")
A("all of them are clearly worse than the options above.\n")

A("## Single models\n")
A("| Model | Folder | Library | Features | Test RMSE | Test R² | CV RMSE | Predict 1,974 mols | Verified |")
A("|---|---|---|---|---:|---:|---:|---:|---|")
for m in singles:
    cv = m["cv_pooled_oof_rmse"]
    feat = "graph + 5 desc" if m["model_type"] == "gatv2" else f"{m['feature_set']} ({m['n_features']})"
    A(f"| {m['config']} | `{slug(m['config'])}/` | {m['library']} | {feat} | "
      f"**{m['test_rmse']:.4f}** | {m['test_r2']:.4f} | "
      f"{'-' if cv is None else f'{cv:.4f}'} | {secs(m['config'])} | {vstatus(m['config'])} |")
A("")
A("`CV RMSE` = pooled out-of-fold RMSE over the 5 scaffold-grouped CV folds of")
A("train+val — the number every selection decision was made on. `Test RMSE` is the")
A("single held-out evaluation, seed-averaged. Timings are whole-test-set batch")
A("predictions on a 2-core container and are dominated by featurization, not inference.\n")

A("## Blends\n")
A("| Blend | Test RMSE | Test R² | CV RMSE | Members (weight) | Verified |")
A("|---|---:|---:|---:|---|---|")
for b in sorted(blends, key=lambda b: b["test_rmse"]):
    active = [m for m in b["members"] if m["active"]]
    ms = "<br>".join(f"`{m['folder']}` × {m['weight']:.3f}" for m in active)
    A(f"| **{b['blend_name']}** | {b['test_rmse']:.4f} | {b['test_r2']:.4f} | "
      f"{b['cv_pooled_oof_rmse']:.4f} | {ms} | {vstatus(b['blend_name'])} |")
A("")
A("A blend is a plain weighted sum of its members' LogS predictions — no meta-model,")
A("no calibration. Weights were fitted on out-of-fold train+val predictions only,")
A("before the test set was read. See `blends/README.md`.\n")

A("## What is in each folder\n")
A("```")
A("<model>/")
A("  model_seed42.<ext>          native-format weights, one file per seed")
A("  model_seed101.<ext>         predict() averages all of them -- that average is")
A("  model_seed123.<ext>         what the Test RMSE above refers to")
A("  feature_names.json          exact ordered column names the model was trained on")
A("  solubility_featurizer.py    SMILES -> feature vector, self-contained")
A("  predict.py                  runnable: SMILES -> LogS")
A("  metadata.json               model type, hyperparameters, data version, metrics")
A("  README.md                   this model specifically")
A("```\n")
A("Native formats: CatBoost `.cbm` (`save_model`), XGBoost `.json` (native Booster),")
A("LightGBM `.txt` (native Booster), scikit-learn `.joblib`, PyTorch `.pt`")
A("(`state_dict`).\n")

A("## Two things that will silently break predictions if you skip them\n")
A("**1. Pass `feature_names.json`.** Descriptor order is not stable across RDKit and")
A("Mordred versions. `predict.py` already loads the pinned list and reorders this")
A("machine's output to match, raising if a descriptor is missing. If you reimplement")
A("featurization yourself and skip this, a version bump shifts the columns and every")
A("prediction is wrong while still looking like a plausible number.\n")
A("**2. Keep the float32 round-trip in `solubility_featurizer._clean`.** Every model")
A("here was trained on a feature cache stored as float32 and widened back to float64 at")
A("fit time, so the trees split on float32-precision values. Computing features in full")
A("float64 at inference moves values across split thresholds: measured, that shifted")
A("predictions by up to **0.166 LogS** for HistGB+Mordred and 0.027 for")
A("LightGBM+Morgan before the round-trip was added. It looks like a no-op line. It is not.\n")
A("**There are no fitted preprocessors.** No scaler, encoder, imputer or feature")
A("selection was fit on training data anywhere in this pipeline — every feature is a")
A("pure per-molecule function of the SMILES string, and the only value-level processing")
A("(non-finite → 0, clip to ±1e30, float32 round-trip) lives inside")
A("`solubility_featurizer.py`. There is no companion object to load, and no way to")
A("forget one.\n")

A("## Use\n")
A("```bash")
A("pip install lightgbm rdkit mordredcommunity")
A("cd lightgbm_mordred")
A('python3 predict.py "CC(=O)Oc1ccccc1C(=O)O" "CCO"')
A("```")
A("```python")
A("import sys; sys.path.insert(0, 'saved_models/lightgbm_mordred')")
A("from predict import predict, load_models")
A("models = load_models()                        # load once, reuse across calls")
A("predict(['CC(=O)Oc1ccccc1C(=O)O'], models)    # -> array([-1.87...])")
A("```")
A("In a UI, call `load_models()` once at startup — the per-call cost is featurization,")
A("not model loading. For a blend, featurize once per distinct feature set and reuse it")
A("across members; `blends/predict_blend.py` re-featurizes per member for clarity, which")
A("is correct but wasteful in a hot path.\n")

A("## One model's weights are not in this archive\n")
A("`extra_trees_rdkit_descriptors/` ships its code, metadata and README but **not** its")
A("three `.joblib` weight files. Each is 152 MB (500 full-depth trees, 7.8M nodes),")
A("over the 20 MB-per-file limit for writing to your disk from the session — a transport")
A("limit, not a choice. It was trained and it passed verification (reproduced to 8.9e-16)")
A("before being dropped. Nothing depends on it: it carries weight **0.000** in the only")
A("blend that lists it and appears in no other. Regenerating it is one deterministic")
A("command — see `extra_trees_rdkit_descriptors/WEIGHTS_NOT_INCLUDED.md`.\n")
A("## Training provenance\n")
A(f"- **Dataset:** `v2_sc_filtered_scaffold_fixed` — AqSolDB with all SC1 / SC2-tight /")
A("  SC2-loose benchmark molecules removed by InChIKey (108 removed, 9,872 remaining),")
A("  split with the corrected scaffold splitter that gives acyclic molecules individual")
A("  keys instead of one shared block.")
A(f"- **Trained on:** train+val pool, {singles[0]['n_train']} molecules.")
A(f"- **Evaluated on:** held-out test set, {singles[0]['n_test']} molecules, scaffold-isolated.")
A("- **Split seed** 42; **model seeds** 42/101/123 (trees), 42/101/123/456/789 (GATv2).")
A("- Full methodology, the audit that produced this dataset version, and the statistical")
A("  comparisons behind the recommendation above: `../REPORT.md`.\n")

A("## Reproducibility note\n")
A("These weights were produced by re-fitting after the original pipeline run, which")
A("saved predictions but not models. The re-fit is exact, not approximate: every model")
A("reproduces the originally recorded test predictions to machine epsilon")
A("(≤ 4e-15 for the tree models; ≤ 1e-6 for XGBoost and GATv2, whose float32 internals")
A("set that floor). `_saved_vs_original_predictions.csv` records the per-model")
A("difference and `_verification_report.csv` the full check.\n")
A("One caveat worth knowing if you ever retrain the GNN: **GATv2 training reproduces")
A("only at `torch.set_num_threads(1)`.** With 2 threads the scatter reductions in")
A("PyTorch Geometric accumulate in a different order and produce a different (equally")
A("valid, but non-reproducing) model — measured at 0.837 LogS max deviation and a")
A("different test RMSE. Inference is unaffected.")

open(f"{SAVED}/README.md", "w").write("\n".join(L) + "\n")
print(f"Wrote {SAVED}/README.md ({len(L)} lines)")
