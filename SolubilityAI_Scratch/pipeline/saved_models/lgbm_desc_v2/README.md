# lgbm_desc_v2

**What it is:** lgbm model on feature set `rdkit_desc`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/lgbm_desc_v2/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/lgbm_desc_v2")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `featurize(smiles, 'rdkit_desc', desc_names)` + descriptors in the exact order of `config.json:desc_names`.

**Preprocessor:** None required (model handles raw features / NaN natively).

**Hyperparameters / config:** see `config.json` (`{"n_estimators": 1500, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 5, "subsample": 0.831877718809044, "subsample_freq": 1, "colsample_bytree": 0.2699753813151923, "reg_lambda": 0.15960727025988383}`); seed(s) 42.
Trained on: data/06b_train_v2.csv (n=7108), dataset v2 (split file data/05b_split_assignments_v2.csv); script `scripts/04_cv_sweep.py`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 0.940 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 0.940 / 0.840 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 0.847 [0.768, 0.933] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.865 [0.849, 0.88] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 0.931 / 0.838 |
| SC2-tight RMSE / R² (logS0, see caveats) | 1.033 / 0.334 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 5.551115123125783e-17, test RMSE reproduced = 0.9308081109166456.

**Files:** README.md, config.json, featurize.py, model.lgbm.txt, performance.json, predict.py, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
