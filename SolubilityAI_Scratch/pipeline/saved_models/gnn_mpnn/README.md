# gnn_mpnn

**What it is:** gnn model on feature set `graph`.

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/gnn_mpnn/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/gnn_mpnn")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** `featurize.py` in this folder: `standardize()` (RDKit parse + rdMolStandardize.Cleanup, counterions kept) then `graphs()` (atom/bond features, see ATOM_LIST/atom_features/bond_features).

**Preprocessor:** None required (model handles raw features / NaN natively).

**Hyperparameters / config:** see `config.json` (`{"n_atom": 40, "n_bond": 7, "hidden": 200, "layers": 4, "n_global": 0, "dropout": 0.1, "head": 256}`); seed(s) [42].
Trained on: data/06_train.csv (n=7365); script `scripts/05_gnn.py gnn_mpnn 0 80 42`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 1.099 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 1.020 / 0.812 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 0.927 [0.838, 1.023] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.839 [0.816, 0.86] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 0.980 / 0.820 |
| SC2-tight RMSE / R² (logS0, see caveats) | 1.165 / 0.153 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 4.440892098500626e-16, test RMSE reproduced = 0.9800874077139526.

**Files:** README.md, __pycache__, config.json, featurize.py, gnn_model.py, model_seed42.pt, performance.json, predict.py, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
