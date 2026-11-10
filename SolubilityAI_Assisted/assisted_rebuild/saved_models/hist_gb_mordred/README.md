# hist_gb[mordred]

scikit-learn HistGradientBoosting over Mordred descriptors. Fully deterministic (identical across seeds).

## What it is
- **Library / model type:** scikit-learn — `hist_gb`
- **Saved format:** `joblib` (joblib.dump), one file per seed: 3 seeds (42, 101, 123)
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
| Test RMSE (seed-averaged) | **0.9775** |
| Test MAE | 0.6622 |
| Test R² | 0.8476 |
| 5-fold scaffold-CV pooled-OOF RMSE | 1.0261 |
| Per-seed test RMSE spread | 0.0 |

Test set = 1974 molecules, scaffold-isolated from training, with all SC1 /
SC2-tight / SC2-loose benchmark compounds already removed from the whole dataset.

## Use it
```bash
pip install scikit-learn rdkit mordredcommunity
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
  "categorical_features": "from_dtype",
  "early_stopping": false,
  "interaction_cst": null,
  "l2_regularization": 1.0,
  "learning_rate": 0.05,
  "loss": "squared_error",
  "max_bins": 255,
  "max_depth": null,
  "max_features": 1.0,
  "max_iter": 800,
  "max_leaf_nodes": 63,
  "min_samples_leaf": 20,
  "monotonic_cst": null,
  "n_iter_no_change": 10,
  "quantile": null,
  "random_state": 42,
  "scoring": "loss",
  "tol": 1e-07,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```
