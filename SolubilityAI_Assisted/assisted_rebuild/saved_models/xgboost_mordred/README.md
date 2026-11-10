# xgboost[mordred]

XGBoost over Mordred 2D descriptors. Nearly as accurate as the LightGBM version and much faster to fit.

## What it is
- **Library / model type:** XGBoost — `xgboost`
- **Saved format:** `json` (Booster.save_model() (native JSON)), one file per seed: 3 seeds (42, 101, 123)
- **Trained on:** v2_sc_filtered_scaffold_fixed — train+val pool, 7898 molecules
- **Split:** scaffold-grouped, seed 42; the 1974-molecule test set below was never seen in training

## Input
Takes a 1613-dimensional vector, feature set `mordred`.

`solubility_featurizer.py` in this folder does the whole SMILES → input conversion.
`feature_names.json` pins the exact column order the model was trained with — pass it
in (as `predict.py` does) so a different RDKit/Mordred version cannot silently shift
the columns.

**There is no fitted preprocessor.** No scaler, encoder, imputer or feature selection
was fit on the training data — every feature is a pure per-molecule function of the
SMILES string, and the only value-level processing is non-finite → 0 and clipping to
±1e30, both already inside `solubility_featurizer.py`. Nothing else to load.

## Output
A single number per molecule: **LogS**, the base-10 logarithm of aqueous solubility in
mol/L. LogS −4 ≈ 10⁻⁴ mol/L. Most oral drugs sit between −4 and −1.

## Performance
| Metric | Value |
|---|---|
| Test RMSE (seed-averaged) | **0.9702** |
| Test MAE | 0.6603 |
| Test R² | 0.8499 |
| 5-fold scaffold-CV pooled-OOF RMSE | 1.0176 |
| Per-seed test RMSE spread | 0.00224 |

Test set = 1974 molecules, scaffold-isolated from training, with all SC1 /
SC2-tight / SC2-loose benchmark compounds already removed from the whole dataset.

## Use it
```bash
pip install xgboost rdkit mordredcommunity
python3 predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
```
```python
from predict import predict
predict(["CC(=O)Oc1ccccc1C(=O)O"])     # -> array([...]) LogS
```
`predict()` loads all 3 seeds and averages them — that average is what the
metrics above describe. A single seed is a slightly different, slightly worse model.

## Hyperparameters
```json
{
  "objective": "reg:squarederror",
  "base_score": null,
  "booster": null,
  "callbacks": null,
  "colsample_bylevel": null,
  "colsample_bynode": null,
  "colsample_bytree": 0.8,
  "device": null,
  "early_stopping_rounds": null,
  "enable_categorical": false,
  "eval_metric": null,
  "feature_types": null,
  "feature_weights": null,
  "gamma": null,
  "grow_policy": null,
  "importance_type": null,
  "interaction_constraints": null,
  "learning_rate": 0.05,
  "max_bin": null,
  "max_cat_threshold": null,
  "max_cat_to_onehot": null,
  "max_delta_step": null,
  "max_depth": 6,
  "max_leaves": null,
  "min_child_weight": null,
  "missing": NaN,
  "monotone_constraints": null,
  "multi_strategy": null,
  "n_estimators": 300,
  "n_jobs": -1,
  "num_parallel_tree": null,
  "random_state": 42,
  "reg_alpha": null,
  "reg_lambda": null,
  "sampling_method": null,
  "scale_pos_weight": null,
  "subsample": 0.8,
  "tree_method": "hist",
  "validate_parameters": null,
  "verbosity": null
}
```
