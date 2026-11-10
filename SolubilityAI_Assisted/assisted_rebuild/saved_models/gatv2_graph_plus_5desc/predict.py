"""
predict.py -- Hybrid GATv2
Predict aqueous solubility (LogS, log10 mol/L) for one or more SMILES strings.

    python3 predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"

Loads all 5 saved seeds and averages. Requires torch and torch_geometric.
No fitted preprocessor: the 17 atom features, 4 bond features and 5 global
descriptors are all computed directly from the molecule.
"""
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gatv2_model import HybridMolecularGATv2, smiles_to_hybrid_graph  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402


def load_models():
    paths = sorted(glob.glob(os.path.join(HERE, "model_seed*.pt")))
    if not paths:
        raise FileNotFoundError("no model_seed*.pt files next to predict.py")
    models = []
    for p in paths:
        m = HybridMolecularGATv2()
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    return models


def predict(smiles_list, models=None):
    graphs = []
    for i, s in enumerate(smiles_list):
        g = smiles_to_hybrid_graph(s)
        if g is None:
            raise ValueError(f"RDKit could not parse SMILES at index {i}: {s!r}")
        graphs.append(g)
    if models is None:
        models = load_models()
    loader = list(DataLoader(graphs, batch_size=256, shuffle=False))
    out = []
    for m in models:
        preds = []
        with torch.no_grad():
            for batch in loader:
                preds.extend(m(batch).squeeze(-1).cpu().numpy().ravel())
        out.append(np.array(preds))
    return np.mean(np.column_stack(out), axis=1)


if __name__ == "__main__":
    smis = sys.argv[1:] or ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    for s, p in zip(smis, predict(smis)):
        print(f"{{s}}\t{{p:.4f}}")
