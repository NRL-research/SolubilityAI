# gnn_hybrid

**What it is:** gnn_hybrid model on feature set `graph+rdkit_desc(quantile-normal)`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/gnn_hybrid/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/gnn_hybrid")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `graphs()` (atom/bond features, see ATOM_LIST/atom_features/bond_features) + descriptors in the exact order of `config.json:desc_names`.

**Preprocessor:** `preprocessor.joblib` (sklearn Pipeline, fitted on the full training set)

**Hyperparameters / config:** see `config.json` (`{"n_atom": 40, "n_bond": 7, "hidden": 200, "layers": 4, "n_global": 223, "dropout": 0.1, "head": 256}`); seed(s) [42].
Trained on: data/06_train.csv (n=7365); script `scripts/05_gnn.py gnn_hybrid 1 80 42`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 1.051 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 0.986 / 0.824 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 0.929 [0.838, 1.03] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.838 [0.813, 0.86] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 0.967 / 0.825 |
| SC2-tight RMSE / R² (logS0, see caveats) | 1.183 / 0.127 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 8.881784197001252e-16, test RMSE reproduced = 0.9666605009361848.

**Files:** README.md, __pycache__, config.json, featurize.py, gnn_model.py, model_seed42.pt, performance.json, predict.py, preprocessor.joblib, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
