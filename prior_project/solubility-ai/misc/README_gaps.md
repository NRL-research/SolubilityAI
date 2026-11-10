---
title: "SolubilityAI — README vs. Repository: Gap Report"
author: "Prepared for review with project collaborator"
date: "2026-08-05"
---

# SolubilityAI — README vs. Repository Gap Report

This document lists everything the project `README.md` describes or claims that is not actually present in the repository (`github.com/avani-jha/solubility-ai`, branch `main`, as of commit `33c1cce`). The goal is to confirm which of these are intentional simplifications (e.g. code that lives elsewhere and hasn't been merged yet) versus items that were dropped or never finished.

---

## 1. `src/` — README claims 5 modules, 1 exists

| File claimed in README | Status |
|---|---|
| `src/splitting.py` (scaffold + random split logic) | **Missing** |
| `src/featurizer.py` (Pillar 1+2 descriptor computation) | **Missing** — closest match is `src/phase1_featurize.py`, a different file at a different path, not wired into any downstream pipeline |
| `src/retrieval.py` (Tanimoto similarity / RAG retrieval) | **Missing** |
| `src/rag_pipeline.py` (full RAG-LLM batched pipeline) | **Missing** |
| `src/evaluate.py` (RMSE / MAE / R² utilities) | **Missing** |
| `src/__init__.py` | Present, but empty (0 bytes) |

## 2. `scripts/` — README claims 7 files, repo has 2 under different names

| File claimed in README | Status |
|---|---|
| `phase0_scaffold_split.py` | **Missing** — repo has `scripts/phase0_canonicalize_and_split.py` instead |
| `phase0b_random_split.py` | **Missing** — repo has `scripts/phase0b_canonicalize_and_random_split.py` instead |
| `phase1_featurize.py` | **Missing from `scripts/`** — exists, but at `src/phase1_featurize.py` |
| `phase2_retrieval_test.py` | **Missing entirely**, no equivalent anywhere in the repo |
| `phase5_batched_rag.py` (described as "Main RAG pipeline (production)") | **Missing entirely**, no equivalent anywhere |
| `evaluate_rag_results.py` (metrics on checkpointed RAG results) | **Missing entirely**, no equivalent anywhere |
| `prepare_dissolvr_data.py` | **Missing from `scripts/`** — exists, but at `dissolvr/prepare_dissolvr_data.py` |

## 3. `models/` — claimed structure

| Item claimed in README | Status |
|---|---|
| `models/holy_trinity/` (GNN + XGBoost + CatBoost weights, gitignored) | Directory **does not exist** |
| `models/dissolvr/` (reproduced DISSOLVR weights, gitignored) | Directory **does not exist** |
| `models/README.md` | Present, but is a single line containing a literal `\n\n` typed as text instead of an actual line break |

## 4. `results/` — entire directory claimed, none of it exists

| Item claimed in README | Status |
|---|---|
| `results/` directory | **Does not exist** |
| `results/rag_results_batched.csv` (checkpointed RAG predictions) | **Missing** |
| `results/metrics_summary.csv` (final metrics across all models) | **Missing** |
| `results/figures/` (plots for paper) | **Missing** |

## 5. "Usage" commands that would fail if run today

| README step | Problem |
|---|---|
| `python scripts/phase0_scaffold_split.py` | Wrong filename (see §2) |
| `python scripts/phase0b_random_split.py` | Wrong filename (see §2) |
| `python scripts/phase1_featurize.py` | File is not in `scripts/` (see §2) |
| `python scripts/phase5_batched_rag.py` | File does not exist |
| `python scripts/evaluate_rag_results.py` | File does not exist |
| `cd dissolvr && python prepare_dissolvr_data.py / train.py / eval_test.py` | Filenames are correct, but every script has a hardcoded absolute Windows path baked in (`C:\Users\aashi\OneDrive\Documents\IIT Dhanbad\...`) — will only run unmodified on that original machine |

## 6. Claimed components with zero supporting code anywhere in git history

- **The entire RAG-LLM pipeline.** No retrieval code, no prompt construction, no Gemini API calls exist anywhere in the repo or its git history — not in `src/`, not in `scripts/`, not in the notebook. The README's "Component 2" architecture diagram (Morgan fingerprints → Tanimoto similarity → prompt construction → Gemini API) and its results-table row (RAG-LLM Pipeline, Scaffold split, RMSE 1.769, R² 0.408, "Run completed") have no supporting artifacts in this repository.
- **"RLCF policy gradients."** The README's architecture diagram states the ensemble is "Blended via Optuna (Bayesian) + RLCF policy gradients." No RLCF or policy-gradient code exists anywhere. Only the Optuna (Bayesian) blending is real, and it lives in `notebooks/solubility_ai.ipynb`.

## 7. Internal inconsistencies between the docs and the actual files

- The main `README.md` directory diagram and `data/README.md` both describe the random-split files as `processed/train_random.csv` / `processed/test_random.csv`. What's actually committed is `data/processed/random_split/train.csv` and `data/processed/random_split/test.csv`. Neither document matches the real path.
- `README.md`'s own repository-structure diagram has a typo: it lists `notebooks/aolubility-ai.ipynb`. The real file is `notebooks/solubility_ai.ipynb`.
- The citation block at the bottom of `README.md` still has an unfilled placeholder: `author = {[authors]}`.

---

## Where the real work actually lives

For context when reviewing the above: essentially all of the actual Holy Trinity ensemble modeling (CNN, GCN, GATv2, Hybrid GATv2, XGBoost/CatBoost partners, Optuna tuning, SHAP analysis, an abandoned ChemBERTa fine-tune) exists only inside `notebooks/solubility_ai.ipynb`, a single ~14,600-line Google Colab notebook committed with full cell outputs. None of it has been extracted into the `src/`/`scripts/` structure the README describes.

The DISSOLVR reproduction (`dissolvr/train.py`, `eval_test.py`, `featurizer.py`, `prepare_dissolvr_data.py`) is real, working code, present at the paths the README implies, though not linked from `scripts/` as documented.
