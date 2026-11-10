# SolubilityAI: Autonomous LLM-Driven Research for Aqueous Solubility Prediction

Code, data, results and headline models accompanying:

> Avani Jha, Dhruv Kumar, Murari Mandal, Naga Rajiv Lakkaniga. *SolubilityAI: Autonomous LLM-Driven Research
> for Aqueous Solubility Prediction.* [CITATION — to be added on publication]

Two autonomous research pipelines, run by the same LLM agent, built aqueous-solubility (LogS) models on AqSolDB
and evaluated them on the Solubility Challenge benchmarks (SC1, SC2-tight, SC2-loose). This repository holds
what each pipeline produced: its code, every intermediate dataset version, all results, the pre-registration
files, and the four headline models reported in the main text.

**Web app.** The four headline models can be used without installing anything at
[SolubilityAI web app](https://solubility-ai.streamlit.app) (source: [<APP-REPO-URL>](<APP-REPO-URL>)).

## Pipeline names

The paper uses formal names. Folders, file names and code keep the working names the pipelines used:

| Name in the paper | Working name | Folder in this repository |
|---|---|---|
| Context-Informed (CI) | Assisted | `SolubilityAI_Assisted/` |
| De Novo (DN) | Scratch | `SolubilityAI_Scratch/` |

## Repository layout

```
prior_project/solubility-ai/        the earlier project repository given to the CI pipeline (snapshot, see below)

SolubilityAI_Assisted/              Context-Informed pipeline
  assisted_rebuild/
    scripts/                        01-15: data foundation, audit, CV, test evaluation, statistics, model saving
    data/                           v1_sc_filtered/ (benchmark overlap removed), audit/, v2_corrected/ (corrected scaffold split, CV folds)
    results/                        CV leaderboard, blend weights, frozen final configs, OOF and test predictions, statistics, bug impact
    round3/                         scripts 20-27 (error analysis, new features and model families, stacking), data/, results/
    saved_models/                   headline models only (see below) + saved-model verification files
  benchmark/
    scripts/                        30-35: validity checks, SC1 integrity, predictions, statistics, analysis, verification
    data/                           SC1 (consistent subset and realigned set), SC2-tight, SC2-loose as evaluated
    results/                        statistics, analyses and per-compound predictions/ for every archived model

SolubilityAI_Scratch/pipeline/      De Novo pipeline
  scripts/                          01-11 (round 1), r2_* (round 2), ext_01-04 (external benchmarks), run_*.sh
  src/                              shared code: standardisation, featurisation, model zoo, CV runner, GNN, predictor template
  data/                             00_raw/ (the supplied input files) and every intermediate dataset version, 01 -> 07d
  results/                          curation and split diagnostics, CV results, OOF predictions, hyperparameter searches,
                                    pre-registration files, test predictions and metrics, verification, external/
  saved_models/                     headline models only (see below) + applicability_domain.py
```

### Input data

Both pipelines received the same four files. They are in `SolubilityAI_Scratch/pipeline/data/00_raw/`:

| File | Content | SHA-256 (first 16) |
|---|---|---|
| `AqSolDB.csv` | AqSolDB, 9,982 entries (ID, Name, InChIKey, SMILES, Solubility) | `81211599ea43ca49` |
| `SC1_processed.csv` | First Solubility Challenge test set, 32 rows | `6d4551c14368d2e4` |
| `SC2_tight_processed.csv` | Second Solubility Challenge, tight set, 100 drugs | `499b313c56e15168` |
| `SC2_loose_processed.csv` | Second Solubility Challenge, loose set, 32 drugs | `d3b2a8befaffdc68` |

Sources: AqSolDB (Sorkun, Khetan & Er, *Sci. Data* 2019, 6, 143); SC1 (Llinas, Glen & Goodman, *J. Chem. Inf.
Model.* 2008, 48, 1289; Hopfinger et al., *J. Chem. Inf. Model.* 2009, 49, 1); SC2 (Llinas, Oprisiu & Avdeef,
*J. Chem. Inf. Model.* 2020, 60, 4791). Please cite the original sources when using these data.

### Prior project (`prior_project/solubility-ai/`)

The CI pipeline was given a local clone of the project's earlier repository. This folder is that clone at commit
`39ed0b674207633d5be0deec12517b725ade5c65` (2026-08-20), without its `.git` folder; its files match that commit
apart from Windows line endings. The pipeline did not modify it. It is included so that readers can see exactly
what code and prior results the CI pipeline inherited (main text Section 2.2; SI Table S31).

One folder of that clone is not included: `dissolvr/`, the earlier project's reproduction of the published DISSOLVR
model (its code and input files). The DISSOLVR code belongs to its authors and is available from them. References to
`dissolvr/` in the snapshot's README, `EXPERIMENTS_LOG.md` and scripts are therefore left as they were, and
`solubility_models/scripts/final_comparison_with_ci.py`, which reruns that code, cannot be run from this
repository. The other prior materials (the project history document, the latest notebook and the code-review
notes) are described in the paper but are not included.

## Headline models

| Pipeline | Model | Folder | Internal test |
|---|---|---|---|
| CI | Seven-model blend | `SolubilityAI_Assisted/assisted_rebuild/saved_models/blends/greedy_blend.json` + its 7 member folders | RMSE 0.9520, R² 0.8555 (n = 1,974) |
| CI | LightGBM, Mordred descriptors | `SolubilityAI_Assisted/assisted_rebuild/saved_models/lightgbm_mordred/` (also a blend member) | RMSE 0.9609, R² 0.8527 |
| DN | Pre-registered NNLS blend | `SolubilityAI_Scratch/pipeline/saved_models/blend_nnls/` + its 12 component folders | RMSE 0.799, R² 0.880 (n = 1,804 in-domain) |
| DN | LightGBM, RDKit + Mordred descriptors | `SolubilityAI_Scratch/pipeline/saved_models/lgbm_desc_mordred/` (also a component) | RMSE 0.832, R² 0.870 |

Every model folder is self-contained: the native model file(s), any fitted preprocessor, the exact featurisation
code, a `predict.py` and a short README. Predicting from SMILES (output: LogS, log10 mol/L):

```bash
pip install -r requirements.txt          # Python 3.11

python SolubilityAI_Assisted/assisted_rebuild/saved_models/blends/predict_blend.py greedy_blend "CC(=O)Oc1ccccc1C(=O)O"
python SolubilityAI_Assisted/assisted_rebuild/saved_models/lightgbm_mordred/predict.py "CC(=O)Oc1ccccc1C(=O)O"
python SolubilityAI_Scratch/pipeline/saved_models/blend_nnls/predict.py "CC(=O)Oc1ccccc1C(=O)O"
python SolubilityAI_Scratch/pipeline/saved_models/lgbm_desc_mordred/predict.py "CC(=O)Oc1ccccc1C(=O)O"
```

Expected output for aspirin: −1.807, −1.823, −1.856 and −1.820. Two details matter for exact reproduction: the
CI GATv2 member reproduces exactly only with one CPU thread, and the CI features are computed at 32-bit precision
(SI, "Reproducibility and Verification"); both are handled inside the saved code. Each folder's
`reference_predictions.csv` (DN) or the CI `_saved_vs_original_predictions.csv` lets you check a fresh install.

`SolubilityAI_Scratch/pipeline/saved_models/applicability_domain.py` reports whether a compound falls inside the
DN applicability domain and its nearest-neighbour Tanimoto similarity to the DN training set.

**Other archived models.** Both pipelines archived more models than the four above (CI: 10 single models and 4
blends; DN: every trained model). Their cross-validation, test and external-benchmark predictions and statistics
are all included under `results/`, so every number in the SI tables can be traced here. Their model files are not
included in this repository; they are available from the corresponding author on request.

## Reproducing the pipelines

Install the pinned environment (`requirements.txt`; Python 3.11.15, CPU only, the versions in SI Table S32).

### De Novo pipeline

Run from `SolubilityAI_Scratch/pipeline/`. First rebuild the one file stored in parts:

```bash
python reassemble_large_files.py         # data/07b_features_mordred.npz, checked against its SHA-256
```

Round 1, as recorded by the pipeline:

```bash
python scripts/01_curate.py              # data/01..04, results/01_curation_log.json
python scripts/02_split.py               # data/05..06, results/02_split_diagnostics.json
python scripts/01b_domain_filter.py      # data/04b, 05b, 06b (in-domain subset, same split)
python scripts/03_features.py            # data/07_features.npz
python scripts/03b_features_mordred.py   # data/07b_features_mordred.npz
bash   scripts/run_tabular.sh            # tabular stages of scripts/04_cv_sweep.py
SOL_DATA=v2 python scripts/04_cv_sweep.py v2_fixed
python scripts/05_gnn.py gnn_mpnn 0 80 42
python scripts/05_gnn.py gnn_hybrid 1 80 42
python scripts/04_cv_sweep.py mordred
python scripts/06_blend.py
python scripts/07_test_eval.py           # the one-time test evaluation (see note)
python scripts/08_verify.py
```

Round 2 runs the `r2_*` scripts in numerical order (`run_round2_tab.sh`, `run_round2_combo.sh`,
`run_round2_gnn.sh` and `run_round2_gnn2.sh` give the exact arguments). The external benchmark evaluation runs
`ext_01` to `ext_04`.

Re-running the training steps recreates every model folder. Running only the evaluation steps on this repository
is limited by the models included: `07` and `08` cover only the model folders present, and `ext_01` stops because it
also evaluates two blends that are not included (`blend_nnls_family`, `r2_blend_nnls_ergfix`). Their recorded
outputs are all in `results/`.

Note: `07_test_eval.py` and `r2_07*_test_eval*.py` refuse to run when their pre-registration file already exists
in `results/`. This is the safeguard that limited each final configuration to a single test evaluation. To
re-run them, work on a copy of the folder and remove the corresponding `*_preregistration.json` there first.

### Context-Informed pipeline

The CI scripts write to the absolute paths of the agent's working environment. Before running them, point these
constants at your copy of the repository (edit them, or create the two directories as links):

| Path in the scripts | Corresponds to |
|---|---|
| `/mnt/user-data/uploads/SolubilityAI_Assisted` (`UPLOADS`, `UP`) | a folder containing the four input files above |
| `/home/claude/work` (`ROOT`) | `SolubilityAI_Assisted/assisted_rebuild/` |
| `/home/claude/work/data`, `.../results` | `assisted_rebuild/data/`, `assisted_rebuild/results/` (round-3 outputs are archived in `assisted_rebuild/round3/`) |
| `/home/claude/work/data/benchmark`, `.../results/benchmark` | `SolubilityAI_Assisted/benchmark/data/`, `.../results/` |
| `/home/claude/work/data/features_cache` | not archived; regenerated by `scripts/04_build_feature_cache.py` (and `round3/scripts/21_build_new_features.py`) |

Order of execution, from `assisted_rebuild/`:

```bash
python scripts/01_sc_leakage_filter.py
python scripts/02_audit_scaffold_split.py
python scripts/03_build_corrected_splits.py
python scripts/04_build_feature_cache.py          # then again with: mordred
python scripts/05_gnn_cv.py cv                    # then: final <seed> for seeds 42 101 123 456 789
python scripts/06_tree_cv.py --baselines          # then the sweep configurations
python scripts/07_assemble_cv_results.py
python scripts/08_final_test_eval.py              # the single test evaluation
python scripts/09_verify.py
python scripts/10_statistical_tests.py
python scripts/11_bug_impact.py
python scripts/12_train_and_save.py               # 12-14: save, reload and verify the archived models
python round3/scripts/20_error_analysis.py        # 20-27 in numerical order (round 3)
python ../benchmark/scripts/30_benchmark_validity.py   # 30-35 in numerical order (external benchmarks)
```

As for the DN pipeline, `12_train_and_save.py` recreates all 14 archived models. Run on this repository as it is,
`14_verify_saved_models.py` and `32_benchmark_predict.py` cover only the models included here.

## Where the paper's numbers are

| Paper item | CI | DN |
|---|---|---|
| Overlap removal and curation (Tables 2, S1–S3) | `assisted_rebuild/data/v1_sc_filtered/` (`01_counts.json`, `sc_removed_molecules.csv`) | `results/01_curation_log.json`, `data/03_removed_sc_overlap.csv`, `data/04_removed_curation.csv` |
| Partitioning (Tables 3, S4–S6) | `assisted_rebuild/data/audit/`, `data/v2_corrected/03_counts.json`, `results/bug_impact.*` | `results/02_split_diagnostics.json` |
| CV and internal test results (Tables 5, S7–S9) | `results/cv_leaderboard.csv`, `results/final_test_results.csv`, `round3/results/` | `results/07_*`, `results/r2_07*`, `results/cv_results.csv` |
| Pre-registration / frozen selection | `results/final_configs.json`, `results/blend_weights.json` | `results/07_preregistration*.json`, `results/r2_07*_preregistration*.json` |
| Paired-bootstrap comparisons (Tables S16–S17) | `results/statistical_comparisons.*`, `round3/results/statistical_comparisons_round3.*` | `results/07_test_metrics.*`, `results/r2_07_verdict.json` |
| External benchmarks (Tables 6, S21–S30) | `benchmark/results/` (`33_stats.*`, `34_*`, `predictions/`) | `results/external/` (`ext_metrics.csv`, `pred_<model>__<set>.csv`) |

## Not included

The pipelines' run logs and their own written reports are not included; the pre-registration and verification
files they rely on are. The instructions given to each pipeline are reproduced verbatim in SI Section S1. The
earlier project's DISSOLVR reproduction (`dissolvr/`) is left out of the snapshot, as explained above.

## License

Code: see `LICENSE`. The input data remain under the terms of their original sources (see "Input data").
