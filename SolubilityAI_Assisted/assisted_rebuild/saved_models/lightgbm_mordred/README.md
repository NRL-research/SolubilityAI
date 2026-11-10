# lightgbm[mordred]

Best single model in the project. LightGBM over ~1,600 Mordred 2D descriptors.

## What it is
- **Library / model type:** LightGBM — `lightgbm`
- **Saved format:** `txt` (Booster.save_model() (native text)), one file per seed: 3 seeds (42, 101, 123)
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
| Test RMSE (seed-averaged) | **0.9609** |
| Test MAE | 0.6515 |
| Test R² | 0.8527 |
| 5-fold scaffold-CV pooled-OOF RMSE | 1.0135 |
| Per-seed test RMSE spread | 0.003028 |

Test set = 1974 molecules, scaffold-isolated from training, with all SC1 /
SC2-tight / SC2-loose benchmark compounds already removed from the whole dataset.

## Use it
```bash
pip install lightgbm rdkit mordredcommunity
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
  "boosting_type": "gbdt",
  "class_weight": null,
  "colsample_bytree": 0.6,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": -1,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 1500,
  "n_jobs": -1,
  "num_leaves": 63,
  "objective": null,
  "random_state": 42,
  "reg_alpha": 0.1,
  "reg_lambda": 1.0,
  "subsample": 0.8,
  "subsample_for_bin": 200000,
  "subsample_freq": 1,
  "verbose": -1
}
```
