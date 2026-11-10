# gatv2[graph+5desc]

The Hybrid GATv2 graph neural network branch of the original SolubilityAI ensemble.

## What it is
- **Library / model type:** PyTorch + PyTorch Geometric — `gatv2`
- **Saved format:** `pt` (torch.save(model.state_dict())), one file per seed: 5 seeds (42, 101, 123, 456, 789)
- **Trained on:** v2_sc_filtered_scaffold_fixed — train+val pool, 7898 molecules
- **Split:** scaffold-grouped, seed 42; the 1974-molecule test set below was never seen in training

## Input
Takes a molecular graph (17 atom features, 4 bond features) plus 5 global descriptors [MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors].

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
| Test RMSE (seed-averaged) | **1.2009** |
| Test MAE | 0.8781 |
| Test R² | 0.7700 |
| 5-fold scaffold-CV pooled-OOF RMSE | 1.1864 |
| Per-seed test RMSE spread | 0.038653 |

Test set = 1974 molecules, scaffold-isolated from training, with all SC1 /
SC2-tight / SC2-loose benchmark compounds already removed from the whole dataset.

## Use it
```bash
pip install torch torch_geometric rdkit
python3 predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
```
```python
from predict import predict
predict(["CC(=O)Oc1ccccc1C(=O)O"])     # -> array([...]) LogS
```
`predict()` loads all 5 seeds and averages them — that average is what the
metrics above describe. A single seed is a slightly different, slightly worse model.

## Hyperparameters
```json
{
  "architecture": "HybridMolecularGATv2",
  "gat1": "GATv2Conv(17 -> 32, heads=4, edge_dim=4)",
  "gat2": "GATv2Conv(128 -> 64, heads=2, edge_dim=4)",
  "head": "Linear(133->64) -> Linear(64->32) -> Linear(32->1)",
  "dropout": [
    0.1,
    0.2
  ],
  "pool": "global_mean_pool",
  "optimizer": "AdamW(lr=0.001, weight_decay=0.01)",
  "loss": "MSELoss",
  "epochs": 60,
  "batch_size": 64,
  "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)"
}
```
