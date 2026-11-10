# xgb_desc_morgan_v2

**What it is:** xgb model on feature set `desc+morgan_counts_r2_2048`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/xgb_desc_morgan_v2/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/xgb_desc_morgan_v2")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `featurize(smiles, 'desc+morgan_counts_r2_2048', desc_names)` + descriptors in the exact order of `config.json:desc_names`.

**Preprocessor:** None required (model handles raw features / NaN natively).

**Hyperparameters / config:** see `config.json` (`{"n_estimators": 1000, "learning_rate": 0.04, "max_depth": 7, "min_child_weight": 1.0, "subsample": 0.7430915675236087, "colsample_bytree": 0.2251629844076013, "reg_lambda": 2.0391192048614384}`); seed(s) 42.
Trained on: data/06b_train_v2.csv (n=7108), dataset v2 (split file data/05b_split_assignments_v2.csv); script `scripts/04_cv_sweep.py`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 0.941 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 0.941 / 0.840 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 0.844 [0.768, 0.928] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.866 [0.85, 0.881] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 0.924 / 0.840 |
| SC2-tight RMSE / R² (logS0, see caveats) | 0.968 / 0.415 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 4.440892098500626e-16, test RMSE reproduced = 0.9235469532848983.

**Files:** README.md, __pycache__, config.json, featurize.py, model.xgb.json, performance.json, predict.py, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
