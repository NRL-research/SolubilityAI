# lgbm_desc_mordred

**What it is:** lgbm model on feature set `desc+mordred`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/lgbm_desc_mordred/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/lgbm_desc_mordred")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `featurize(smiles, 'desc+mordred', desc_names)` + descriptors in the exact order of `config.json:desc_names`.

**Preprocessor:** None required (model handles raw features / NaN natively).

**Hyperparameters / config:** see `config.json` (`{"n_estimators": 1500, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 5, "subsample": 0.831877718809044, "subsample_freq": 1, "colsample_bytree": 0.2699753813151923, "reg_lambda": 0.15960727025988383}`); seed(s) 42.
Trained on: data/06_train.csv (n=7365), dataset v1 (split file data/05_split_assignments.csv); script `scripts/04_cv_sweep.py`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 0.979 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 0.923 / 0.846 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 0.832 [0.756, 0.917] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.870 [0.854, 0.884] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 0.869 / 0.859 |
| SC2-tight RMSE / R² (logS0, see caveats) | 1.019 / 0.353 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 4.440892098500626e-16, test RMSE reproduced = 0.8691671136827914.

**Files:** README.md, __pycache__, config.json, featurize.py, model.lgbm.txt, performance.json, predict.py, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
