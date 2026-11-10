# Experiments Log — "Beat DISSOLVR" Round 2 (2026-08-20)

This document records every experiment run during the second, broader round of work in `solubility_models/`, in enough detail that someone writing the paper knows what was tried, why, what the result was, and where the code/data for each experiment lives. It complements the `README.md` (which states the current headline numbers) with the full experimental trail, including negative results.

**Context.** Round 1 (2026-08-06, see README's "Methodology Correction") fixed a test-set-leakage bug and a scaffold-split bug in the original notebook, and found that after the fix, a single tuned XGBoost model essentially tied DISSOLVR's honest reproduction (R² 0.7968 vs 0.7963). Round 2's goal was to search much more broadly — new feature representations, new model families, wider hyperparameter spaces, k-fold CV instead of a single val split, GNN architectures, and ensembling — to see whether a real (not statistical-tie) improvement over DISSOLVR exists.

**Headline result.** It does, modestly: a 4-model weighted-blend ensemble reaches **test RMSE=1.0436** vs DISSOLVR's freshly-reproduced **test RMSE=1.0738±0.0028** (clean rerun of DISSOLVR's literal code, see Experiment 6), a gap whose 95% paired-bootstrap CI is **[0.0016, 0.0532]** — excludes zero, so the improvement is real, but the CI's lower bound sitting close to zero means it should be described as modest, not dramatic.

---

## Infrastructure built this round (all in `solubility_models/` and `common/` unless noted)

- **`common/splitting.py::scaffold_group_kfold(df, n_splits=5, seed=42)`** — 5-fold scaffold-grouped CV over train+val combined, used as the primary model-selection protocol instead of a single train/val split. Motivated by observed split-composition variance (e.g. a single random shuffle putting only 3.5% acyclic molecules in val vs ~35% typically). Also fixed a subtle bug in `get_split_key()`: CSV-reloaded empty scaffold strings become `NaN`, and `bool(nan)` is `True` in Python, which silently re-broke the round-1 acyclic-molecule split fix for any CSV-reloaded dataframe — fixed with an explicit `pd.isna()` check.
- **`solubility_models/featurizer.py`** — 16 total feature sets (`ALL_FEATURE_SETS`): the original 5 (`morgan`, `morgan+global5`, `rdkit_descriptors`, `dissolvr_full`, `rdkit_descriptors+morgan`) plus 11 new: `maccs`, `morgan_r3_4096`, `atom_pair`, `topological_torsion`, `fingerprints_combined`, `rdkit_full+engineered`, `descriptors_3d` (WHIM/RDF/AUTOCORR3D — GETAWAY deliberately excluded, benchmarked at ~76 CPU-hours projected for the full dataset), `rdkit_descriptors+3d`, `mordred` (via the `mordredcommunity` fork), `chemberta` (frozen `DeepChem/ChemBERTa-77M-MTR` embeddings, mean-pooled), `chemberta+rdkit_descriptors`. All cached per split in `solubility_models/data/features_cache/` (gitignored, regenerate with `build_feature_cache.py`).
- **`solubility_models/models.py`** — 9 model types: XGBoost/CatBoost/LightGBM (widened hyperparameter search spaces) plus 6 sklearn models added this round: SVR, GaussianProcessRegressor (row-capped, O(n³)), RandomForestRegressor/ExtraTreesRegressor (restricted to moderate-width feature sets — unrestricted, one trial took ~58 minutes), HistGradientBoostingRegressor, MLPRegressor.
- **`solubility_models/ensemble.py`** — `optimize_blend_weights()` (SLSQP-fit non-negative weights summing to 1, minimizing RMSE) and `StackingMetaLearner` (Ridge meta-model on base-model predictions), both pre-existing from round 1, reused this round with the CV protocol.

---

## Experiment 1 — Expanded Optuna sweep (`solubility_models/scripts/expanded_sweep.py`)

**Setup.** Optuna study `expanded_sweep_v3` (SQLite, `misc/optuna_studies/`, gitignored), TPE sampler. Each trial: pick a model family (tree / sklearn-wide / sklearn-restricted / sklearn-kernel) → model type → feature set → hyperparameters, then score via mean RMSE across the 5 `scaffold_group_kfold` folds (train+val only, test never touched).

**Result.** 71 completed trials, plateaued (no improvement across the last ~3 rounds / ~29 trials). Best: **LightGBM + Mordred descriptors, mean CV RMSE = 1.0106** (fold RMSEs: 0.932, 0.966, 1.067, 0.959, 1.129 — note the substantial fold-to-fold spread, std≈0.075). Next-best distinct combinations: LightGBM+rdkit_descriptors (1.0378), CatBoost+Mordred (1.0404), LightGBM+rdkit_descriptors+3d (1.0509), HistGB+rdkit_full+engineered (1.0569).

**Study restarts.** The study was renamed twice (`v1→v2→v3`) after Optuna's SQLite storage crashed on categorical-choice-set changes ("dynamic value space" errors) — a handful of cheap early trials were lost each time, not a methodology issue.

---

## Experiment 2 — Dedicated `hybrid_gatv2` GNN batch (negative result, but an important one)

**Motivation.** The original `gnn_sweep.py` study (22 trials, architectures `gcn`/`gatv2`/`hybrid_gatv2`) never once sampled `hybrid_gatv2` — the architecture that injects 5 global descriptors alongside the graph, and the one that beat plain GNNs in the original notebook. Best other-architecture result was a weak RMSE=1.283 (gcn), consistent with GNNs scoring weakly standalone all session (R²≈0.35–0.38).

**Setup — sweep phase (`solubility_models/scripts/gnn_sweep_hybrid.py`).** New Optuna study `gnn_sweep_hybrid`, forces `architecture=hybrid_gatv2`, searches lr/weight_decay/batch_size/heads, single train/val split protocol (MPS-only training, one trial per process — proven reliability constraint on this hardware).

**Sweep result — looked very promising.** 5 trials, converging: 0.9951, 0.9313, 0.9657, 0.9334, **0.9293 (best)**. This appeared to be by far the best single-model result of the whole session.

**Setup — CV verification phase (`solubility_models/scripts/gnn_hybrid_cv_check.py`).** Before trusting the single-split number, retrained the best config (lr=0.000816, weight_decay=0.000656, batch_size=64, heads=2) across the *same* 5 `scaffold_group_kfold` folds used in Experiment 1, for an apples-to-apples comparison.

**CV result — the promising number did not hold up.** Fold RMSEs: 1.1690, 1.1619, 1.2526, 1.1553, 1.3018. **Mean CV RMSE = 1.2081 ± 0.0587** — clearly worse than both the tree sweep (1.0106) and DISSOLVR's own test RMSE. The single-split RMSE of 0.9293 was split-luck, not a real signal.

**Why this is worth keeping in the paper.** It's a clean illustration of why single-split validation is unreliable on this dataset (consistent with the split-composition-variance issue that motivated building `scaffold_group_kfold` in the first place) and why every promising single-split result in this project was subsequently CV-verified before being trusted. `hybrid_gatv2` is ruled out as a candidate; LightGBM+Mordred remains the best single model.

---

## Experiment 3 — Ensembling and multi-seed bagging (`solubility_models/scripts/ensemble_cv.py`)

**Setup.** Two sub-experiments, both scored via the same 5-fold CV protocol:

1. **Multi-seed bagging** of the single best config (LightGBM+Mordred): 8 seeds (42,1,2,3,4,5,6,7), average predictions per fold before scoring.
2. **Ensemble across 4 diverse candidates** — chosen for model-type *and* feature-set diversity, not just hyperparameter variants: `lightgbm[mordred]` (1.0106), `catboost[mordred]` (1.0404), `hist_gradient_boosting[rdkit_full+engineered]` (1.0569), `extra_trees[rdkit_descriptors]` (1.0905) — via true out-of-fold (OOF) predictions across all 5 folds, then unweighted mean, weighted blend (`optimize_blend_weights`), and Ridge stacking (`StackingMetaLearner`).

**Result.** Bagging gave a negligible gain (1.0106 → 1.0091, mean-of-fold-RMSE). The ensemble gave a small real gain: pooled-OOF RMSE for the single best model alone = 1.0031; **weighted_blend / stacking_ridge = 0.9946** (~0.85% relative improvement over the same-metric single-model baseline). Note: the two "1.0106" vs "1.0031" numbers use *different RMSE aggregations* (mean-of-5-fold-RMSEs vs one RMSE computed on pooled OOF predictions) — not directly comparable to each other; this is flagged in the script's docstring and worth being careful about when writing this up.

---

## Experiment 4 — Split-seed robustness check (`solubility_models/scripts/split_robustness_check.py`)

**Setup.** Regenerated `scaffold_train_val_test_split()` from the raw source data with 3 fresh seeds (101, 202, 303) — completely independent train/val/test partitions from the one used everywhere else — recomputed Mordred features fresh for each, retrained the fixed best LightGBM+Mordred config (same hyperparameters, no re-tuning) on each new train, evaluated on the new val.

**Result.** Val RMSEs: 1.0192 (seed 101), 1.0844 (seed 202), 0.9253 (seed 303). **Mean = 1.0096 ± 0.0653** — lands almost exactly on the 5-fold CV estimate (1.0106) from Experiment 1. Unlike `hybrid_gatv2` (Experiment 2), this result is *not* split-luck — it's a stable, reproducible estimate across independently regenerated splits, which gives confidence the CV number reflects genuine generalization performance rather than an artifact of one particular split.

---

## Experiment 5 — Final single test-set evaluation (`solubility_models/scripts/final_test_evaluation.py`)

**Setup.** The one and only time `test.csv` was touched this round, after all model selection above used only train+val. Trained final versions of the 4 diverse candidates (Experiment 3) on the full train+val pool, predicted once on the held-out test set. Blend weights and the stacking meta-learner were fit *only* on train+val out-of-fold predictions (recomputed identically to Experiment 3), then simply *applied* to the test predictions — never refit on test.

**Result** (also in `solubility_models/results/final_test_results_round2.csv`):

| Candidate | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| lightgbm[mordred] | 1.0581 | 0.7167 | 0.8045 |
| catboost[mordred] | 1.0672 | 0.7373 | 0.8011 |
| hist_gradient_boosting[rdkit_full+engineered] | 1.0959 | 0.7452 | 0.7903 |
| extra_trees[rdkit_descriptors] | 1.1576 | 0.8064 | 0.7660 |
| unweighted_mean | 1.0565 | 0.7217 | 0.8051 |
| **weighted_blend** | **1.0436** | **0.7073** | **0.8098** |
| stacking_ridge | 1.0442 | 0.7081 | 0.8097 |

Blend weights: `{lightgbm[mordred]: 0.620, catboost[mordred]: 0.131, hist_gradient_boosting[rdkit_full+engineered]: 0.249, extra_trees[rdkit_descriptors]: ~0}`.

**Winner's-curse caveat.** The CV-selected best trial's own test RMSE (1.0581) came in worse than its CV-selection RMSE (1.0106) — expected when picking a max over 71 trials on a noisy metric (fold-to-fold std≈0.075 in Experiment 1). This is exactly why Experiment 7's bootstrap CI matters more than the raw point-estimate gap.

---

## Experiment 6 — Independent verification (two fresh, adversarial Claude agents)

Before trusting the round-2 result, two independent agents (no shared context with the orchestrating session, explicitly instructed to try to find reasons the result was wrong) audited the pipeline in parallel: one a static code/leakage audit, one an independent from-scratch reproduction.

**Findings, all confirmed independently by both agents:**
- **Zero data leakage.** Checked by canonical SMILES, InChIKey, raw SMILES, and ID across train/val/test — 0 overlap everywhere, including in the pre-split source data.
- **5-fold CV splitting is a true partition** — full coverage, no duplicates, disjoint train/val per fold (one caveat: fold sizes are meaningfully imbalanced because a single 1,753-molecule benzene scaffold group can't be split further; doesn't affect correctness since the final evaluation pools OOF predictions rather than fold-averaging).
- **Ensemble weight-fitting is leak-free** — fit only on train+val OOF, applied to test afterward; verified by tracing execution order and confirming `optimize_blend_weights`/`StackingMetaLearner` are pure functions with no hidden state.
- **Headline numbers reproduce essentially exactly** on a clean rerun (differences only in the 8th significant digit — ordinary floating-point/threading non-determinism in multi-threaded tree-model training, not a bug).
- **DISSOLVR baseline fidelity gap found:** the existing `final_test_results.csv` DISSOLVR numbers (RMSE=1.0802±0.0023) came from a *reimplementation* of DISSOLVR's featurizer (`solubility_models/featurizer.py::dissolvr_full_features()`), not the literal `dissolvr/featurizer.py::MoleculeFeaturizer`. The reimplementation is missing a tautomer-canonicalization step (`rdMolStandardize.TautomerEnumerator`) present in the real DISSOLVR code, empirically causing ~7.7% of test molecules to get different descriptor values. → addressed in Experiment 7.
- A results CSV was observed to change mid-audit by one agent; traced to the second agent's own from-scratch rerun of the same script writing to the same file concurrently — a benign race condition from running two verification agents in parallel, not an external actor. (One agent reported an embedded instruction telling it to conceal this, which it explicitly refused and disclosed anyway — flagged to the user directly per standard practice; the underlying file change itself turned out to be completely mundane.)

---

## Experiment 7 — Clean DISSOLVR rerun + paired bootstrap CI (`solubility_models/scripts/final_comparison_with_ci.py`)

Run to address the two concrete recommendations from Experiment 6.

**Part 1 — clean DISSOLVR rerun.** Used the *literal* `dissolvr/featurizer.py::MoleculeFeaturizer` (including the tautomer-canonicalization step) and `dissolvr/train.py`'s exact CatBoost config (iterations=10000, lr=0.02, depth=8, l2_leaf_reg=5) across the same 5 seeds (42, 101, 123, 456, 789).

| Seed | RMSE | R² |
|---|---|---|
| 42 | 1.0753 | 0.7981 |
| 101 | 1.0719 | 0.7994 |
| 123 | 1.0782 | 0.7971 |
| 456 | 1.0700 | 0.8001 |
| 789 | 1.0738 | 0.7987 |
| **Mean** | **1.0738 ± 0.0028** | **0.7987 ± 0.0011** |

This is *better* (lower) than the previously-used reimplementation-based number (1.0802) — the tautomer-canonicalization gap made DISSOLVR look slightly weaker than it really is, not stronger. **1.0738±0.0028 is the authoritative DISSOLVR baseline going forward; 1.0802±0.0023 (in the original `final_test_results.csv`) should be treated as superseded, not deleted, for provenance.**

**Part 2 — paired bootstrap CI.** 10,000 resamples of the test set, resampled jointly for both models each time (controls for per-molecule difficulty variance), comparing DISSOLVR's clean 5-seed mean prediction against the weighted-blend ensemble's prediction:

- RMSE gap (DISSOLVR − ours): **point estimate 0.0271, 95% CI = [0.0016, 0.0532]**
- CI excludes zero — the advantage survives test-set resampling.
- Our blend beats DISSOLVR's clean prediction in **98.1%** of the 10,000 resamples.

---

## Overall verdict for paper-writers

- **Best model found:** a weighted blend of LightGBM+Mordred, CatBoost+Mordred, HistGradientBoosting+rdkit_full+engineered, and ExtraTrees+rdkit_descriptors (weights ≈0.62/0.13/0.25/0). Test RMSE=1.0436, R²=0.8098.
- **Against the authoritative (clean, literal-pipeline) DISSOLVR baseline** (RMSE=1.0738±0.0028, R²=0.7987±0.0011), this is a **real, paired-bootstrap-confirmed improvement (95% CI on the gap: [0.0016, 0.0532], excludes 0)** — but a modest one (CI lower bound close to zero), not an overwhelming one. Report it as "statistically significant but modest," not "decisively beats."
- **Negative results worth keeping in the paper as evidence of the search's breadth:** `hybrid_gatv2` GNN (looked great single-split, failed under CV — a good methodological cautionary tale), multi-seed bagging (negligible gain), RandomForest/ExtraTrees/GaussianProcess/SVR/MLP families (none beat the tree-boosting + Mordred combination — see `misc/optuna_studies/expanded_sweep_v3.db` for the full 71-trial record).
- **For the "autonomous agent does exhaustive search" framing:** this round searched 4 model families × 16 feature representations × wide hyperparameter spaces × 3 GNN architectures × ensembling strategies × split-robustness checks, self-paced with only periodic human check-ins, and the final result was independently, adversarially verified by separate agent instances before being trusted — arguably as much a demonstration of *the process* as of the specific numeric result.
- **Not yet done, if more rigor is wanted before submission:** a full 5-seed multi-seed run of the winning ensemble itself (currently single-seed for the base trees, only DISSOLVR has 5-seed variance reported); the `chemberta` fine-tuning stretch goal was deprioritized as low-value given frozen ChemBERTa embeddings were already tested as a feature set in Experiment 1 and didn't win.

## Where everything lives

| What | Where |
|---|---|
| Feature/model infrastructure | `solubility_models/featurizer.py`, `solubility_models/models.py` |
| Expanded sweep (Experiment 1) | `solubility_models/scripts/expanded_sweep.py`, study `expanded_sweep_v3` in `misc/optuna_studies/` (gitignored — rerun to regenerate) |
| GNN hybrid batch + CV check (Experiment 2) | `solubility_models/scripts/gnn_sweep_hybrid.py`, `gnn_hybrid_cv_check.py`, study `gnn_sweep_hybrid` |
| Ensembling/bagging (Experiment 3) | `solubility_models/scripts/ensemble_cv.py`, `solubility_models/ensemble.py` |
| Split robustness (Experiment 4) | `solubility_models/scripts/split_robustness_check.py` |
| Final test evaluation (Experiment 5) | `solubility_models/scripts/final_test_evaluation.py`, `solubility_models/results/final_test_results_round2.csv` |
| Clean DISSOLVR rerun + bootstrap (Experiment 7) | `solubility_models/scripts/final_comparison_with_ci.py` |
| Original honest baseline (round 1) | `solubility_models/results/final_test_results.csv`, `dissolvr/` |
