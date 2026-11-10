# SolubilityAI

**SolubilityAI: A Scaffold-Aware Machine Learning Framework for Aqueous Solubility Prediction**

---

## Abstract

SolubilityAI predicts aqueous solubility (LogS) of drug-like molecules from SMILES representations, built around a **scaffold-aware evaluation protocol** that prevents structural leakage between training and test sets. The project's core goal is to beat a reproduction of the published **DISSOLVR** model with tree/GNN approaches on the same honest split. A Retrieval-Augmented LLM pipeline (Gemini, grounded by structural analogues) also exists as a separate, currently deprioritized exploration.

**A 2026-08-06 correction is central to this project's actual finding**: an earlier version of the scaffold-aware evaluation was itself compromised by test-set leakage during hyperparameter search, which inflated a "Holy Trinity" GNN+tree ensemble's reported advantage over DISSOLVR. After fixing that leakage and an unrelated scaffold-split bug, honest tuning shows a single well-tuned XGBoost model essentially ties DISSOLVR's reproduction — see [Key Results](#key-results) and [Methodology Correction](#methodology-correction-2026-08-06). The RAG-LLM pipeline was implemented separately on 2026-08-10 but has only been run against the original, pre-correction test split — its result isn't yet a like-for-like comparison with the corrected numbers, and it's on hold while effort focuses on the tree/GNN approach.

**2026-08-19: the repository was reorganized** around that focus — see [Repository Structure](#repository-structure).

**2026-08-20: a broader round of search (new feature representations, model families, GNN architectures, ensembling) found a real, independently-verified improvement over DISSOLVR** — see [Key Results](#key-results) and `EXPERIMENTS_LOG.md` for the full experimental trail, including negative results and the verification process.

---

## Key Results

**Updated 2026-08-20.** A second, broader experimentation round searched 4 model families × 16 feature representations × wide hyperparameter spaces × 3 GNN architectures × ensembling strategies, all model-selected via 5-fold scaffold-grouped cross-validation (never touching test), then independently, adversarially verified by two fresh agent instances before being trusted. Full detail, including every negative result, in `EXPERIMENTS_LOG.md`.

The DISSOLVR baseline below is a **clean rerun of the literal `dissolvr/` code** (not a reimplementation — see `EXPERIMENTS_LOG.md` Experiment 7 for why that distinction mattered) on the corrected scaffold split. The new candidate is a weighted blend of 4 tree models, model-selected on train+val only and evaluated exactly once on the held-out test set.

| Model | RMSE | R² | Notes |
|---|---|---|---|
| DISSOLVR (reproduced, clean rerun of literal code) | 1.0738 ± 0.0028 | 0.7987 ± 0.0011 | 5 seeds. **Corrected split, test.** Supersedes the 1.0802±0.0023 figure below (that number used a featurizer reimplementation missing a tautomer-canonicalization step). |
| **Weighted-blend ensemble** (LightGBM+Mordred / CatBoost+Mordred / HistGB+rdkit_full+engineered / ExtraTrees+rdkit_descriptors) | **1.0436** | **0.8098** | Blend weights fit on train+val CV only. **Corrected split, test.** |

> **Finding:** the blend beats the clean DISSOLVR baseline by RMSE 1.0436 vs 1.0738. A paired bootstrap (10,000 resamples on the same test molecules) puts the 95% CI on that gap at **[0.0016, 0.0532]** — excludes zero, so the improvement is real, not test-set noise, but the CI's lower bound sitting close to zero means it's a **modest**, not decisive, margin. A dedicated `hybrid_gatv2` GNN batch initially looked like the best single model of the whole project (single-split val RMSE=0.9293) but its true 5-fold CV RMSE was 1.2081±0.0587 — worse than everything else — a useful cautionary result on why every promising single-split number in this project gets CV-verified before being trusted. See `EXPERIMENTS_LOG.md` for the complete list of what was tried, including model families and feature sets that did *not* help.

<details>
<summary>2026-08-10 figures (round 1) — superseded by the above, kept for the record</summary>

The first two rows are evaluated on the corrected scaffold split (80/20, Murcko scaffold, `common/data/processed_scaffold_fixed/`) — validation was used for every tuning decision, and the test set was touched exactly once per model, after tuning was finalized. The RAG-LLM row is evaluated on the **original, pre-correction** split (`misc/original_buggy_split/`) and is not directly comparable to the other two.

| Model | RMSE | R² | Notes |
|---|---|---|---|
| DISSOLVR (reproduced), untuned | 1.0802 ± 0.0023 | 0.7963 ± 0.0009 | 5 seeds, original published hyperparameters, no tuning. **Corrected split, test.** Superseded by the clean rerun above (1.0738±0.0028). |
| Tuned XGBoost (585-trial Optuna sweep, validation-only) | 1.0789 | 0.7968 | RDKit descriptor features, single seed. **Corrected split, test.** |
| RAG-LLM Pipeline | 1.769 | 0.408 | Gemini 2.5 Flash-Lite, k=10 retrieval, 1,996/1,996 molecules. **Original (pre-correction) split.** |

**Round-1 finding (now superseded):** after fixing the test-set leakage and a scaffold-split bug, the tuned model's edge over DISSOLVR's reproduction essentially disappeared: R² 0.7968 vs 0.7963, a gap smaller than DISSOLVR's own seed-to-seed variance — a statistical tie, not yet a real win. Round 2 (above) found a real, if modest, margin instead.

</details>

<details>
<summary>Original (superseded) figures — kept for the record, do not cite</summary>

| Model | Split | RMSE | R² | Notes |
|---|---|---|---|---|
| Holy Trinity Ensemble | Random | 0.9986 | 0.8235 | Baseline (inflated) |
| Holy Trinity Ensemble | Scaffold | 0.9756 | 0.8202 | **Invalidated** — hyperparameters and blend weights were chosen by 50+ rounds of Optuna search scored directly against this test set |
| DISSOLVR (reproduced) | Random | 0.8158 ± 0.0030 | 0.8624 ± 0.0010 | From published figures, not independently reproduced |
| DISSOLVR (reproduced) | Scaffold | 0.9910 ± 0.0028 | 0.8145 ± 0.0011 | Run on the original (buggy) scaffold split — see below |

</details>

---

## Methodology Correction (2026-08-06)

A review of `misc/notebooks/solubility_ai.ipynb` found the original scaffold-split results above were not trustworthy, for two independent reasons:

1. **Test-set leakage in hyperparameter search.** Every Optuna study in the notebook (cells 31, 42, 54) computed its optimization objective — including the final 50-trial "Holy Trinity" search — directly against the test set's R². Both tree models' hyperparameters and the ensemble blend weights were chosen specifically to maximize test-set performance. This inflates the reported R²=0.8202 by an unknown amount; DISSOLVR's reproduction, by contrast, used fixed, never-tuned hyperparameters, so the two numbers were never a fair comparison.
2. **Scaffold-split bug.** RDKit's Murcko scaffold is empty for acyclic (ring-free) molecules. The original split grouped every such molecule under one shared key, and because whole scaffold groups are assigned to train or test together, all 2,940 acyclic molecules (29.5% of the dataset) ended up in train and none in test.

Both are fixed in `common/splitting.py`: acyclic molecules now get individual split keys instead of one shared block, and a proper train/validation/test split (`common/data/processed_scaffold_fixed/`) keeps validation for every tuning decision, with test touched exactly once for the final numbers above. The original split and results remain at `misc/original_buggy_split/` for the record, and are marked invalidated rather than deleted.

As a sanity check, the corrected test RMSE (~1.08) now sits comfortably above the ~0.6–0.8 aleatoric noise floor described below — the original RMSE (~0.97–0.99) sitting close to that floor, in hindsight, was itself a leakage red flag.

The RAG-LLM pipeline described in the architecture diagram below had no supporting code as of this correction (2026-08-06). It was added on 2026-08-10 (now at `misc/rag_llm/`) and run against the *original* scaffold split, not the corrected one above — see Key Results for why that number isn't directly comparable yet.

---

## Architecture

```
SMILES Input
    │
    ├─── [Primary] Tree/GNN sweep (solubility_models/) ─────────────────────┐
    │        ├── XGBoost / CatBoost / LightGBM (RDKit descriptors)          │
    │        ├── GCN / GATv2 / Hybrid GATv2 (graph topology)                 ├─► LogS
    │        └── Blend weights + stacking fit on VALIDATION only             │
    │            (NOT Optuna-against-test-set, and NOT "RLCF" -- neither    │
    │             the original blend method nor RLCF policy gradients ever  │
    │             existed as working code; see Methodology Correction)      │
    │                                                                        │
    ├─── [On hold] RAG-LLM Pipeline (misc/rag_llm/) ─────────────────────────┤
    │        ├── Morgan Fingerprints (radius=2, 2048 bit)                   │
    │        ├── Tanimoto similarity → top-k training neighbours            ├─► LogS
    │        ├── Prompt construction (SMILES + LogS + descriptors)          │  (implemented
    │        └── Gemini 2.5 Flash-Lite → JSON prediction                    │   2026-08-10,
    │                                                                        │   deprioritized)
    │                                                                        │
    └─── [Baseline] DISSOLVR Reproduction (dissolvr/) ──────────────────────┘
             ├── 176-feature descriptor vector (3 pillars)
             └── CatBoost (reproduced from ICML 2026 paper)
```

---

## Repository Structure

**Reorganized 2026-08-19** around the project's actual goal: beat DISSOLVR with the tree/GNN approach. Four top-level areas — `common/` (shared by everything), `dissolvr/` (the baseline to beat), `solubility_models/` (where that work happens), and `misc/` (everything else: RAG-LLM, the original notebook, and the now-superseded original split, all kept for the record).

```
solubility-ai/
│
├── README.md
├── EXPERIMENTS_LOG.md               ← full round-2 experimental trail: every experiment, result, and negative finding
├── requirements.txt
├── .gitignore
│
├── common/                          ← shared by dissolvr/ and solubility_models/
│   ├── __init__.py
│   ├── splitting.py                 ← scaffold + random split logic (fixed acyclic-molecule grouping bug)
│   ├── evaluate.py                  ← RMSE / MAE / R² + multi-seed aggregation utilities
│   ├── rebuild_scaffold_split.py    ← generates the splits below from misc/original_buggy_split/
│   └── data/
│       ├── README.md                ← AqSolDB download instructions
│       ├── processed_scaffold_fixed/  ← THE split: train.csv / val.csv / test.csv
│       └── processed_random_fixed/    ← corrected random split (inflated-baseline comparison)
│
├── dissolvr/                        ← DISSOLVR reproduction — the baseline to beat
│   ├── featurizer.py                ← DISSOLVR's own 176-feature computation
│   ├── train.py                     ← CatBoost, 5 seeds (now uses relative paths)
│   ├── eval_test.py
│   ├── prepare_dissolvr_data.py     ← sources from common/data/processed_scaffold_fixed/
│   ├── data/                        ← DISSOLVR-format train.csv / test.csv (regenerated from the corrected split)
│   └── model/                       ← saved model.joblib (gitignored)
│
├── solubility_models/                ← our approaches to beat DISSOLVR — main focus going forward
│   ├── __init__.py                  ← import-order fix for a macOS xgboost/catboost/lightgbm + torch OpenMP conflict
│   ├── featurizer.py                ← Morgan FP / RDKit descriptors / DISSOLVR's engineered features / GNN graph builder
│   ├── models.py                    ← tree model builders (XGBoost/CatBoost/LightGBM) + GNN architectures (GCN/GATv2/Hybrid GATv2)
│   ├── ensemble.py                  ← validation-scored blend weight optimization + stacking meta-learner
│   ├── data/features_cache/         ← cached feature matrices per split x representation (gitignored)
│   ├── scripts/
│   │   ├── build_feature_cache.py   ← precomputes all 16 feature sets x split, cached to data/features_cache/ (gitignored)
│   │   ├── baselines.py             ← honest, untuned DISSOLVR + tree/GNN baselines (validation)
│   │   ├── tree_sweep.py            ← round 1: Optuna sweep, single val split (validation)
│   │   ├── gnn_sweep.py             ← round 1: Optuna sweep, GNN architecture x optimizer hyperparameters (validation)
│   │   ├── ensemble_search.py       ← round 1: combines top sweep results into blend/stacking candidates (validation)
│   │   ├── final_test_eval.py       ← round 1: the ONE test-set evaluation
│   │   ├── expanded_sweep.py        ← round 2: 4 model families x 16 feature sets, 5-fold scaffold CV (train+val only)
│   │   ├── gnn_sweep_hybrid.py      ← round 2: dedicated hybrid_gatv2 Optuna batch (single val split)
│   │   ├── gnn_hybrid_cv_check.py   ← round 2: CV-verifies the above -- found the single-split result was split-luck
│   │   ├── ensemble_cv.py           ← round 2: multi-seed bagging + diverse-candidate ensembling, 5-fold CV
│   │   ├── split_robustness_check.py ← round 2: re-verifies the best config across 3 fresh scaffold-split seeds
│   │   ├── final_test_evaluation.py ← round 2: the ONE test-set evaluation for the round-2 candidates
│   │   └── final_comparison_with_ci.py ← round 2: clean literal-DISSOLVR rerun + paired bootstrap CI vs the winning blend
│   └── results/
│       ├── baselines.csv
│       ├── leaderboard.csv
│       ├── final_test_results.csv         ← round 1 test results
│       └── final_test_results_round2.csv  ← round 2 test results (current best)
│
└── misc/                            ← everything not directly on the DISSOLVR-beating path
    ├── README_gaps.md / .pdf        ← 2026-08-06 audit of README claims vs. actual repo contents
    ├── notebooks/
    │   └── solubility_ai.ipynb      ← original exploratory notebook (source of the Holy Trinity work; contains the now-fixed leakage bug)
    ├── optuna_studies/               ← SQLite-backed, resumable Optuna studies (gitignored)
    ├── original_buggy_split/         ← the invalidated split + the scripts that generated it, kept for provenance
    │   ├── phase0_canonicalize_and_split.py
    │   ├── phase0b_canonicalize_and_random_split.py
    │   └── data_processed/           ← train.csv / test.csv / random_split/
    └── rag_llm/                      ← deprioritized RAG-LLM pipeline (real, working, not on the corrected split yet)
        ├── phase1_featurize.py       ← Pillar 1+2 featurizer + Morgan FP
        ├── phase2_retrieval.py       ← Tanimoto similarity retrieval
        ├── phase5_batched_rag.py     ← main pipeline (Gemini, checkpointed, resumable)
        ├── evaluate_rag_results.py
        └── results/rag_results_batched.csv
```

`common/splitting.py`/`evaluate.py` are the two pieces of preprocessing/eval logic genuinely shared between DISSOLVR and our approach; each side's *feature computation* stays separate since DISSOLVR's is a fixed, specific 176-feature set (kept authentic to the paper reproduction) while `solubility_models/featurizer.py` offers several swappable representations for the sweep — including its own reimplementation of DISSOLVR's feature set (`dissolvr_full`), which is intentionally a separate copy from `dissolvr/featurizer.py` rather than a shared import, so the sweep doesn't depend on `dissolvr/`'s standalone-script structure.

---

## Installation

### Prerequisites

- Python 3.9+
- Anaconda or Miniconda (recommended)
- RDKit **must** be installed via conda, not pip (pip breaks Cairo bindings)

```bash
# 1. Clone the repository
git clone https://github.com/avani-jha/solubility-ai.git
cd solubility-ai

# 2. Create and activate conda environment
conda create -n solubility-ai python=3.10
conda activate solubility-ai

# 3. Install RDKit via conda (critical — do not use pip for RDKit)
conda install -c conda-forge rdkit

# 4. Install all other dependencies
pip install -r requirements.txt
```

### API Keys (for the on-hold RAG-LLM pipeline)

Set environment variables before running RAG scripts:

```bash
# Windows
set GOOGLE_API_KEY=your_gemini_api_key_here

# Linux / macOS
export GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## Data

AqSolDB is not bundled in this repository due to size. Download it from the official source:

- **AqSolDB**: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8)
- Place the downloaded CSV at `common/data/raw/AqSolDB.csv`

Pre-processed splits are committed for reproducibility: the original (buggy, invalidated) splits at `misc/original_buggy_split/data_processed/`, and the corrected train/val/test splits actually used for the results above at `common/data/processed_scaffold_fixed/` and `common/data/processed_random_fixed/`.

---

## Usage

### Step 1 — Generate the corrected train/val/test splits

```bash
python common/rebuild_scaffold_split.py
# -> common/data/processed_scaffold_fixed/{train,val,test}.csv
# -> common/data/processed_random_fixed/{train,val,test}.csv
```

`misc/original_buggy_split/phase0_canonicalize_and_split.py` / `phase0b_canonicalize_and_random_split.py` regenerate the original, buggy split — kept for reproducing the superseded figures, not recommended for new work.

### Step 2 — Precompute and cache features (tree/GNN pipeline)

```bash
python solubility_models/scripts/build_feature_cache.py
```

### Step 3 — Honest baselines, then the automated sweep

```bash
python solubility_models/scripts/baselines.py                                  # untuned DISSOLVR + Holy Trinity, validation
python solubility_models/scripts/tree_sweep.py --n-trials 1000 --timeout 480   # Optuna, feature set x model x hyperparams
python solubility_models/scripts/gnn_sweep.py --n-trials 1                     # one process per trial — see note below
python solubility_models/scripts/ensemble_search.py --top-k 5                  # blend/stacking candidates -> solubility_models/results/leaderboard.csv
```

`gnn_sweep.py` trains via PyTorch's MPS (Apple GPU) backend; running more than one trial per process has been unreliable on this hardware/torch version, so it's invoked once per trial and re-run in a loop rather than with a large `--n-trials`. Don't run it concurrently with `tree_sweep.py` — that's caused a kill from resource contention.

### Step 4 — Final test-set evaluation (run once, at the end)

```bash
python solubility_models/scripts/final_test_eval.py
```

Evaluates the already-decided best configuration(s) on the held-out test set. Not a step to iterate on — re-running it after seeing its output and changing the model choice defeats the point of holding test out.

### DISSOLVR reproduction

```bash
cd dissolvr
python prepare_dissolvr_data.py    # sources common/data/processed_scaffold_fixed/, writes dissolvr/data/{train,test}.csv
python train.py                    # trains DISSOLVR, 5 seeds, saves the best-by-test-RMSE model to dissolvr/model/
python eval_test.py                # evaluates the saved model on dissolvr/data/test.csv
```

All paths are relative now — this runs from a fresh clone with no editing required, unlike before 2026-08-19.

### RAG-LLM pipeline (on hold)

```bash
cd misc/rag_llm
python phase1_featurize.py       # generates train_featurized.csv / test_featurized.csv (edit INPUT_DIR first)
python phase2_retrieval.py       # sanity-check retrieval on one molecule
# Set GOOGLE_API_KEY first (see Installation)
python phase5_batched_rag.py     # main run — checkpoints after every batch, resumes automatically if interrupted
python evaluate_rag_results.py   # metrics on results/rag_results_batched.csv
```

Currently runs against the original (pre-correction) split — see the note in Key Results. `phase1_featurize.py` and `phase5_batched_rag.py` both have a hardcoded `INPUT_DIR` for the original development machine; edit before running elsewhere.

---

## Methodology Notes

### Why scaffold split?

Random splits allow molecules with nearly identical ring skeletons (scaffolds) to appear in both training and test sets. This is a form of data leakage: the model effectively memorizes structural families during training and then "predicts" test molecules it has already seen close cousins of. Scaffold splits force entire scaffold families into either train or test, requiring genuine generalization to new structural chemotypes.

### Pillar 1 + 2 Descriptors (RAG context)

Retrieved neighbours are presented with:
- **Pillar 1 (Functional Group Identity):** `fr_*` fragment counts from RDKit — ketones, esters, amines, halogens, phenols, etc.
- **Pillar 2 (Topology):** BalabanJ, BertzCT, Kappa1/2, ring counts, rotatable bond count

We deliberately excluded Pillar 3 (Joback melting point + Abraham solvation proxies) because these are heuristic approximations of thermodynamic quantities, not directly computed from SMILES, and add approximate noise rather than clean signal for the LLM context.

### Aleatoric Limit

The experimental noise floor for aqueous solubility (inter-laboratory variability) is estimated at ~0.6–0.8 RMSE logS units across multiple independent studies. Any model reporting RMSE below this range warrants scrutiny for data leakage or non-standard curation.

## Current Status

| Component | Status |
|---|---|
| Repository reorganized around beating DISSOLVR | Complete — `common/`, `dissolvr/`, `solubility_models/`, `misc/` |
| Corrected scaffold split + random split (train/val/test) | Complete — `common/data/processed_scaffold_fixed/`, `processed_random_fixed/` |
| DISSOLVR reproduction, portable (relative paths, sources the corrected split) | Complete |
| Validation-only Optuna sweep (tree models + GNN architecture) | Complete — 585+ tree trials + GNN architecture trials, see `solubility_models/results/leaderboard.csv` |
| Final single test-set evaluation (tree/GNN side, round 1) | Complete — `solubility_models/results/final_test_results.csv` |
| Round 2: expanded CV sweep (16 feature sets × 4 model families) | Complete — 71 trials, plateaued, see `EXPERIMENTS_LOG.md` |
| Round 2: dedicated `hybrid_gatv2` GNN batch | Complete — negative result (CV RMSE=1.2081, worse than tree sweep) |
| Round 2: ensembling + multi-seed bagging | Complete — weighted blend, small real gain |
| Round 2: split-seed robustness check | Complete — result stable across 3 independent splits |
| Round 2: independent adversarial verification (2 agents) | Complete — no leakage found; DISSOLVR baseline refreshed |
| Round 2: final test evaluation + paired bootstrap CI | Complete — `solubility_models/results/final_test_results_round2.csv` |
| **Beat DISSOLVR by a real margin** | **Achieved (modest) — RMSE 1.0436 vs clean baseline 1.0738±0.0028, 95% bootstrap CI on the gap excludes 0** |
| Holy Trinity Ensemble (original notebook) | Complete, but results invalidated by test-set leakage — see Methodology Correction |
| RAG-LLM pipeline (Gemini, k=10, batch=20) | On hold — 1,996 / 1,996 molecules run, but on the original (pre-correction) split |
| ChemBERTa fine-tune (vs. frozen embeddings already tested as a feature set) | Deprioritized — frozen embeddings didn't win in the round-2 sweep |
| SHAP interpretability layer | Planned |

---

## Citation

If you use this code or findings in your work, please cite:

```bibtex
@misc{solubilityai2026,
  title   = {SolubilityAI: A Scaffold-Aware Machine Learning Framework for Aqueous Solubility Prediction},
  author  = {[authors]},
  year    = {2026},
  note    = {IIT Dhanbad. GitHub: https://github.com/avani-jha/solubility-ai}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgements

- AqSolDB dataset: Sorkun et al. (2019), *Scientific Data*
- DISSOLVR (reproduced): Ramani et al. (2026), *ICML*
- RDKit: Open-source cheminformatics toolkit
