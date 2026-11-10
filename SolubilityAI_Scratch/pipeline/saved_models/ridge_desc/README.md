# ridge_desc

**What it is:** ridge model on feature set `rdkit_desc`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/ridge_desc/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/ridge_desc")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `featurize(smiles, 'rdkit_desc', desc_names)` + descriptors in the exact order of `config.json:desc_names`.

**Preprocessor:** `preprocessor.joblib` (sklearn Pipeline, fitted on the full training set)

**Hyperparameters / config:** see `config.json` (`{"alpha": 10.0}`); seed(s) 42.
Trained on: data/06_train.csv (n=7365), dataset version data/04_aqsoldb_curated.csv; script `scripts/04_cv_sweep.py`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 2.235 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 1.183 / 0.747 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 1.088 [0.997, 1.186] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.778 [0.756, 0.797] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 1.204 / 0.729 |
| SC2-tight RMSE / R² (logS0, see caveats) | 1.202 / 0.098 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 2.220446049250313e-16, test RMSE reproduced = 1.2036124170088265.

**Files:** README.md, __pycache__, config.json, featurize.py, model.joblib, performance.json, predict.py, preprocessor.joblib, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
