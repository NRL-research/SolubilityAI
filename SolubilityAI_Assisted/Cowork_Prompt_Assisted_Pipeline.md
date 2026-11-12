# SolubilityAI — Assisted Autoresearch Pipeline

## Context and Materials Provided

You are continuing an existing machine learning research project on aqueous solubility (LogS) prediction. You have been given:

1. **AqSolDB (AqSolDB.csv)** — the raw, original dataset (\~9,982 molecules, columns: ID, Name, InChIKey, SMILES, Solubility)
2. **SC1, SC2-tight, and SC2-loose processed datasets** — external benchmark compound sets (Hopfinger et al. 2009; Llinás, Oprisiu \& Avdeef 2020), each already resolved to canonical SMILES with InChIKeys
3. **SolubilityAI Project History document (SolubilityAI History 2026-09-14.docx)** — a full narrative record of everything attempted so far: the original ensemble architecture, the RAG-LLM pipeline, the leakage-control rebuild, and all diagnostic experiments and their outcomes
4. **a local clone of the GitHub repository provided as files (solubility-ai)** — all code, processed data splits, and prior results committed so far
5. **The most recent Colab notebook (solubility\_ai\_v2.ipynb)** — containing the latest round of experiments and ablations run on this project.

**Read all of this materials before writing or running any code.** The History document and the Colab notebook may describe different states of the project — the History document is more recent where the two disagree. If you find any other disagreement between what a document claims and what the code/repository actually contains, flag it explicitly rather than silently trusting one source.

Separately, you have also been given access to a second, independently produced set of materials from a project collaborator: a README and an experiments log describing an independent review and correction of this same codebase. **Read these as well before starting.** They describe two specific, verified bugs in the project's prior work:

* A **test-set leakage bug** in hyperparameter and blend-weight tuning (search objectives were scored directly against the test set rather than a held-out validation set)
* A **scaffold-splitting bug affecting acyclic molecules** (molecules with no ring system produce an empty scaffold value; the splitting logic grouped all such molecules into one shared group and placed that entire group into training, removing roughly 29.5% of the dataset from ever appearing in any test set)

## Immediate First Task (Do This Before Anything Else)

**Produce an SC-leakage-filtered version of AqSolDB with all three benchmark sets (SC1, SC2-tight, SC2-loose) removed**, using InChIKey matching, following the same method already established in this project (see History document, Section 8). Save this as a clearly named, standalone CSV output and confirm the exact molecule count removed and remaining. **Deliver this file as soon as it is ready, before proceeding with the rest of this task** — it is needed immediately for a separate, parallel piece of work outside this pipeline.

## Primary Objective

Maximize R² and minimize RMSE for LogS prediction on a genuinely held-out, scaffold-split test set, using AqSolDB (SC-leakage-filtered, per above) as the core dataset. You are the primary owner of this codebase going forward — you are expected to build on, correct, and extend the existing work, not merely reproduce it.

**DISSOLVR is no longer a target for direct experimental comparison.** Do not spend time reproducing DISSOLVR's dataset filtering (molecular weight cutoffs, LogS range restrictions, mixture removal) or matching its exact training conditions. DISSOLVR will be referenced only narratively in the eventual paper's Discussion section, not as a benchmark this pipeline needs to match or beat. Do not treat DISSOLVR's published or reproduced numbers as a target.

## Required Steps

1. **Audit the existing codebase and prior results for correctness before building on them.** Specifically:

   * Investigate whether the acyclic-molecule scaffold-splitting bug described in the collaborator's materials is present in this project's own scaffold-splitting code. Determine its scope and impact on this project's own prior reported results.
   * Investigate whether the test-set leakage pattern described in the collaborator's materials (hyperparameter/blend-weight tuning scored against the test set) is present anywhere in this project's own tuning code, beyond what has already been documented as fixed in the History document.
   * Decide, based on what you find, whether and how to fix each issue. You have discretion here — but any decision to fix, partially fix, or not fix a discovered issue must be stated explicitly and justified in your final report, not left implicit.
2. **Establish a corrected, reproducible data foundation** that combines: SC1/SC2/SC2-loose leakage removal (per the Immediate First Task) with a correct scaffold-splitting implementation (i.e., one that does not mishandle acyclic molecules, if that bug is confirmed present). Document exactly how this combined, corrected dataset differs from every dataset version used in prior work, with molecule counts at each step.
3. **Re-establish honest baseline numbers** on this corrected foundation for the existing SolubilityAI ensemble (GATv2 + XGBoost + CatBoost) and for a solo CatBoost model (the diagnostic comparison already established in prior work — see History document), before attempting any new modeling approaches. This gives you a trustworthy starting point to improve upon.
4. **Continue building on and exploring beyond what has already been tried.** You are not limited to re-running prior experiments — actively look for opportunities to improve on the existing approaches (e.g., the feature representations, model families, and ensembling strategies introduced in the collaborator's Round 2 search, and anything else you judge promising), while keeping the corrected, leakage-free foundation established in steps 1–3 as the basis for every subsequent experiment.
5. **Use proper cross-validation (not a single train/validation split) to select any model, hyperparameter configuration, or ensemble weighting**, and touch the true held-out test set only once per final configuration being reported. This project has direct prior evidence (see History document, and the collaborator's materials, GNN cross-validation finding) that single-split results can be seriously misleading — treat any promising single-split result as provisional until cross-validated.
6. **Verify all final claims independently before reporting them.** At minimum: confirm no molecule appears in both a training and evaluation role at any stage (by molecular identity, not just by index), confirm cross-validation folds are genuinely non-overlapping and scaffold-respecting, and confirm that any reported comparison between two models' accuracy is supported by a statistical test (e.g., a paired bootstrap over the test set) rather than a single point-estimate difference.

## Reproducibility Requirements

* Fix and record random seeds for every stochastic step (data splitting, model initialization, cross-validation fold assignment).
* Every result reported in your final output must be traceable to a specific, saved script or notebook cell, a specific dataset version, and a specific set of hyperparameters — not just a number in a table.
* Save all intermediate dataset versions (raw, SC-filtered, corrected-split) as distinct, clearly named files rather than overwriting in place, so any step can be audited or re-run independently.
* Save incremental progress as clearly named local files/commits for me to push manually.

## Final Deliverable

A written report covering: what was audited and found in the existing codebase (including explicit statements on the two known bugs and their disposition), the final corrected dataset methodology, every model/approach attempted with honest cross-validated results, the single best final configuration with test-set performance and its statistical confidence, and a clear, itemized account of what remains uncertain, unverified, or worth further investigation.

# Additional Task: External Benchmark Evaluation (SC1, SC2-tight, SC2-loose)

Add this as a required part of the work and as a new section of the Word report. It uses the three processed external benchmark datasets you were given. No model should be trained or tuned for this task. It evaluates models that already exist.

## Goal
For each headline model, generate predictions for the compounds in SC1, SC2-tight and SC2-loose, compare them with the experimentally measured values, and compute accuracy statistics for each set. This is an independent check on how well the models generalize to real, externally measured compounds.

## Which models to evaluate
- Every headline model: the 2-3 best models you identified as most important, including any final blend or ensemble.
- The individual component models of any headline blend.

Load each model fresh from its saved files (model file, preprocessor and featurization code), not from memory left over from training. If a headline model was never saved in a reloadable form, say so explicitly and list it as not evaluated. Do not retrain it silently.

## Step 1: Verify the evaluation is valid before running it
These checks decide whether the results mean anything.
1. Confirm, by InChIKey (not by SMILES string), that no SC1, SC2-tight or SC2-loose compound appears in the training data of any model being evaluated. If any overlap exists for a model, report it, exclude that model from this evaluation, and do not present its results as external validation.
2. Confirm these sets were not used for model selection, hyperparameter tuning, early stopping, blend-weight fitting or any other decision. If they were used anywhere, state where and what effect that has.
3. Check the target column units before computing anything. Open each file and look at the actual values of the experimental solubility column. The SC1 file's solubility column is named with a "uM" suffix, but its values may already be log-scale (for example, values like -2.7 rather than values in the thousands). Decide from the actual values, state your reasoning in the report, and apply no unit conversion unless the data genuinely requires it. A wrong conversion will silently corrupt every statistic.
4. Report the number of compounds in each set. If a count differs from the published size of that benchmark set, or if any compound fails SMILES parsing or featurization, list the affected compounds and report the final n. Never drop compounds silently.

## Step 2: Generate predictions
- Predict each set separately. Never pool SC1, SC2-tight and SC2-loose, because they differ in measurement quality and origin.
- Save a per-compound results table for each model and set, with: compound name (if available), InChIKey, canonical SMILES, experimental LogS, predicted LogS and signed error (predicted minus experimental). Add the measurement standard deviation where the file provides one.
- Sanity-check the outputs. Predictions should differ across models and across sets. If different models return identical numbers, or the same numbers appear across sets, treat that as a bug and investigate it before reporting anything.

## Step 3: Compute statistics, per model and per set
- RMSE, MAE and R². Note that R² on sets this small is unstable, so interpret it with care.
- Mean signed error (bias), to show whether the model systematically over- or under-predicts.
- Pearson and Spearman correlation between predicted and experimental values.
- The fraction of compounds predicted within 0.5 and within 1.0 LogS units of the experimental value.
- A bootstrap 95% confidence interval for RMSE and R² on each set, resampling compounds. State the number of resamples and the seed.
- Where the data include per-compound measurement uncertainty, put the model errors in that context. For example, state how much of the SC2-loose error is plausibly explained by that set's higher measurement noise.

## Step 4: Compare and interpret
- Compare each model's benchmark performance with its performance on your own held-out test set. State plainly whether it generalizes comparably, degrades, or improves, and offer possible reasons, labelled as hypotheses and not as findings.
- List the worst-predicted compounds in each set, and note any structural pattern among them.
- State clearly that this is one fixed evaluation. Do not re-tune, retrain or swap models after seeing these results in order to improve them. If you think a change is warranted, describe it as a recommendation without applying it.

## Reporting
- Add a section to the Word report titled "External Benchmark Evaluation". Describe how the validity checks above were done and what they found, then give one results table per benchmark set (model, n, RMSE, MAE, R², bias, correlations, and bootstrap confidence intervals), and a short interpretation. Put full per-model tables for non-headline models in the appendix.
- Add the headline models' benchmark results to the report's main results table or a clearly linked one.
- Save all per-compound prediction files and the evaluation script as separate, clearly named files in the working folder, so every number in the report can be traced and re-run.
- Do not present these results as more certain than they are. State n for every set, and note where the sample size limits what can be concluded.
