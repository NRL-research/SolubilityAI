# svr_desc_v2

**What it is:** svr model on feature set `rdkit_desc`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/svr_desc_v2/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/svr_desc_v2")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `featurize(smiles, 'rdkit_desc', desc_names)` + descriptors in the exact order of `config.json:desc_names`.

**Preprocessor:** `preprocessor.joblib` (sklearn Pipeline, fitted on the full training set)

**Hyperparameters / config:** see `config.json` (`{"C": 11.563857936940023, "gamma": 0.0019849125636726005, "epsilon": 0.05}`); seed(s) 42.
Trained on: data/06b_train_v2.csv (n=7108), dataset v2 (split file data/05b_split_assignments_v2.csv); script `scripts/04_cv_sweep.py`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 0.990 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 0.990 / 0.823 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 0.893 [0.813, 0.98] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.850 [0.832, 0.867] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 0.959 / 0.828 |
| SC2-tight RMSE / R² (logS0, see caveats) | 1.036 / 0.331 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 4.440892098500626e-16, test RMSE reproduced = 0.9585352784067677.

**Files:** README.md, __pycache__, config.json, featurize.py, model.joblib, performance.json, predict.py, preprocessor.joblib, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
