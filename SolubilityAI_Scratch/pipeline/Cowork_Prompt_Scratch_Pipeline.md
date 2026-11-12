# SolubilityAI — Scratch Autoresearch Pipeline

## Context and Materials Provided

You are starting a new machine learning research effort on aqueous solubility (LogS) prediction. You have been given only:

1. **AqSolDB** **(AqSolDB.csv)** — the raw, original dataset (\~9,982 molecules, columns: ID, Name, InChIKey, SMILES, Solubility)
2. **SC1, SC2-tight, and SC2-loose processed datasets** — external benchmark compound sets (Hopfinger et al. 2009; Llinás, Oprisiu \& Avdeef 2020), each already resolved to canonical SMILES with InChIKeys

You have **no access** to any prior codebase, notebook, results, or project history for this effort. This is intentional: this pipeline is a deliberate scratch-built counterpart to a separate, parallel effort that does build on prior work, and its value depends on genuine independence. Do not attempt to locate, infer, or reconstruct any prior work on this dataset from outside sources; approach this as an original research problem.

## Primary Objective

Build the most accurate possible model for predicting LogS from molecular structure (SMILES), maximizing R² and minimizing RMSE on a genuinely held-out test set. You have full latitude to choose your own dataset curation approach, featurization strategy, model families, and evaluation protocol — there is no existing methodology you are required to follow or match.

**DISSOLVR is not a target for direct experimental comparison.** You do not need to reproduce any published model's dataset filtering or training conditions. If you are aware of DISSOLVR or related published solubility-prediction work, you may use it as general background context, but do not treat matching or exceeding any specific published number as a goal of this pipeline.

## Required Steps

1. **Establish your own data foundation.** At minimum, remove any molecule from AqSolDB that also appears in SC1, SC2-tight, or SC2-loose (by molecular identity, e.g. InChIKey matching — not just by exact SMILES string match, since the same molecule can be written as more than one valid SMILES) before using AqSolDB for training. This is required because SC1/SC2 may later be used as an external validation set, and any model trained on molecules that also appear in that validation set would not be genuinely tested on unseen chemistry. Beyond this requirement, you have full discretion over any further curation (handling of duplicates, salts/mixtures, outlier values, etc.) — document and justify whatever choices you make.
2. **Design your own train/test split methodology.** You are free to choose your own approach, but be aware that a naive random split of molecules tends to substantially inflate reported accuracy relative to how a model will perform on genuinely novel chemistry, because structurally near-identical molecules can end up on both sides of a random split. Whatever splitting approach you choose, verify it does not have this or a similar failure mode — in particular, if you use any form of structure-based grouping to construct your split, explicitly check that molecules with no ring system (which can behave differently than typical structure-grouping logic expects) are being correctly and fairly distributed between training and test, not systematically excluded from one side.
3. **Explore featurization and modeling approaches independently.** Consider a broad range of options, which may include (but are not limited to) molecular fingerprints, computed physicochemical descriptors, graph-based neural network approaches, and pretrained chemical language model embeddings. Justify your choices based on your own reasoning about the problem, not by assuming any particular approach is correct.
4. **Use proper cross-validation (not a single train/validation split) to select any model, hyperparameter configuration, or ensemble weighting**, touching the true held-out test set only once per final configuration being reported. Be specifically cautious of promising results obtained from only a single validation split — verify any strong result under cross-validation before trusting or reporting it, since single-split results can be misleadingly optimistic or pessimistic depending on which molecules happen to land in that one split.
5. **Verify all final claims independently before reporting them.** At minimum: confirm no molecule appears in both a training and evaluation role at any stage (by molecular identity, not just by index or exact string match), confirm cross-validation folds are genuinely non-overlapping, and support any comparison between two of your own candidate models with a statistical test (e.g., a paired bootstrap over the test set) rather than a single point-estimate difference.

## Reproducibility Requirements

* Fix and record random seeds for every stochastic step (data splitting, model initialization, cross-validation fold assignment).
* Every result reported in your final output must be traceable to a specific, saved script or notebook cell, a specific dataset version, and a specific set of hyperparameters — not just a number in a table.
* Save all intermediate dataset versions as distinct, clearly named files rather than overwriting in place, so any step can be audited or re-run independently.
* Structure your code and outputs as if a third party will need to reproduce your exact pipeline from scratch using only your saved files and a written record of your steps.



# Additional Requirement: Save Trained Models for Future UI Integration



Add this as a required, ongoing part of the pipeline's work — not a final cleanup step done only at the end, but something maintained continuously as models are trained throughout the session.



## Goal

Every distinct model configuration that is trained and evaluated during this work must be saved in a form that can be reloaded later and used to generate a prediction for a new molecule, without needing to re-run any training. The end goal is that a UI could later be built on top of whichever saved model(s) are selected, so what gets saved must be genuinely self-contained and independently usable.



## Requirements



**Save as you go, not only at the end.** Each time a model is trained and evaluated, save it immediately as part of that step, rather than deferring model-saving to a final wrap-up. If a model is later discarded in favor of something better, keep its saved files rather than deleting them — a full record of what was tried, not just the final answer, is part of the deliverable.



**Save every distinct model, not just the best one.** This includes individual component models that are part of a larger ensemble or blend — if a final result combines several models, save each individual model separately as well as the combination itself.



**For every model saved, include:**

* The trained model file itself, in the standard saved format for whichever library produced it (e.g., a model saved via its library's own native save method — do not invent an ad hoc or generic format).
* If the model is part of a blend or ensemble, the exact combination weights and logic needed to reproduce the final blended prediction from the individual models' outputs, saved as a small, explicit config file.
* Any fitted preprocessing object the model depends on to work correctly on new input (e.g., a fitted scaler or encoder) — a model saved without its matching preprocessor will silently produce wrong predictions later, so this is mandatory, not optional.
* The exact, runnable featurization code needed to go from a raw SMILES string to that specific model's expected input format (e.g., fingerprint type and size, descriptor list and order) — described precisely enough, with actual code, that predictions can be reproduced exactly, not approximately.



**Organize into a clear folder structure.** Create a dedicated folder (e.g. `saved\\\_models/`) with one clearly named subfolder per model. Each subfolder should contain everything needed to use that model independently: the model file, any preprocessor file, the featurization code, and a short model-specific README stating what the model is, what it expects as input, what it outputs, and its recorded performance.



**Write a top-level model index.** At the root of `saved\\\_models/`, maintain one master README listing every saved model, its subfolder location, a one-line description, its key performance metric(s), and a clear, reasoned recommendation of which model(s) are the best candidates to build a UI on top of — not just a flat list with no guidance.



**Verify every saved model actually reloads and works.** For every model saved, run a short verification: load the saved model and any preprocessor from disk in a fresh state (not reusing anything still in memory from training), run it on a small set of known molecules, and confirm the output matches the originally recorded performance within a small numerical tolerance. Do not report a model as successfully saved unless this check has passed; report explicitly and clearly if any model fails this check rather than omitting it.



**If a model cannot be saved for some reason** (e.g., its weights are no longer available in memory or on disk by the time this is addressed), state this clearly and list it as unrecoverable rather than silently skipping it or attempting to retrain it without flagging that this happened.



## Final Deliverable

A written report covering: your chosen data curation and splitting methodology and why you chose it, every model/approach attempted with honest cross-validated results, the single best final configuration with test-set performance and its statistical confidence, and a clear, itemized account of what remains uncertain, unverified, or worth further investigation.


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