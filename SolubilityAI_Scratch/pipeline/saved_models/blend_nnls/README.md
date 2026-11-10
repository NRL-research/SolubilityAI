# blend_nnls

**What it is:** Weighted blend of saved models: `ridge_desc` x 0.019, `knn_morganbits_k5` x 0.001, `svr_desc` x 0.061, `mlp_desc` x 0.034, `lgbm_desc_maccs` x 0.099, `lgbm_desc_v2` x 0.013, `xgb_desc_morgan_v2` x 0.035, `svr_desc_v2` x 0.035, `gnn_mpnn` x 0.191, `gnn_hybrid` x 0.142, `lgbm_mordred` x 0.204, `lgbm_desc_mordred` x 0.184 + intercept 0.0412.  Loading this model loads each component folder (siblings in saved_models/).

**Input:** SMILES string(s).  **Output:** predicted LogS = log10(aqueous solubility, mol/L). Unparseable SMILES -> NaN.

**Use:**
```bash
python predict.py "CCO" "c1ccccc1"
```
```python
import importlib.util
spec = importlib.util.spec_from_file_location("p", "saved_models/blend_nnls/predict.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
predict = m.load_predictor("saved_models/blend_nnls")
predict(["CCO"])  # -> numpy array of LogS
```

**Featurization:** Each component featurizes the SMILES with its own featurize.py.

**Preprocessor:** None of its own (each component carries its own preprocessor).

**Hyperparameters / config:** see `config.json` (`null`); seed(s) None.
Trained on: blend of the component models (weights fitted on in-domain train OOF predictions); script `scripts/06_blend.py`.

**Recorded performance**
| metric | value |
|---|---|
| CV RMSE, own training set (pooled OOF, 5-fold grouped) | 0.887 |
| CV RMSE / R², in-domain train molecules (comparable across models) | 0.887 / 0.858 |
| In-domain test RMSE [95% CI, cluster bootstrap] (n=1804) | 0.799 [0.723, 0.884] |
| In-domain test R² [95% CI, molecule bootstrap] | 0.880 [0.864, 0.895] |
| Full curated test RMSE / R² (n=1852, incl. metal/mixture entries) | 0.840 / 0.868 |
| SC2-tight RMSE / R² (logS0, see caveats) | 0.983 / 0.397 |

**Reload verification (fresh process):** PASS — reference max |Δ| = 8.881784197001252e-16, test RMSE reproduced = 0.8399279494793701.

**Files:** README.md, config.json, featurize.py, performance.json, predict.py, reference_predictions.csv

Requires the package versions in `config.json:versions` (RDKit version matters for descriptor values).
