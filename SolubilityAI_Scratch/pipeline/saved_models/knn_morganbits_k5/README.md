# knn_morganbits_k5

**What it is:** knn model on feature set `morgan_bits_r2_2048`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/knn_morganbits_k5/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/knn_morganbits_k5")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `featurize(smiles, 'morgan_bits_r2_2048', desc_names)`.

**Preprocessor:** None required (model handles raw features / NaN natively).

**Hyperparameters / config:** see `config.json` (`{"k": 5}`); seed(s) 42.
Trained on: data/06_train.csv (n=7365), dataset version data/04_aqsoldb_curated.csv; script `scripts/04_cv_sweep.py`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 1.721 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 1.682 / 0.490 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 1.605 [1.462, 1.761] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.516 [0.464, 0.562] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 1.639 / 0.497 |
| SC2-tight RMSE / R² (logS0, see caveats) | 1.474 / -0.356 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 0.0, test RMSE reproduced = 1.6385528253647423.

**Files:** README.md, __pycache__, config.json, featurize.py, model.joblib, performance.json, predict.py, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
