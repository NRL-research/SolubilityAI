"""
gnn.py
======
The Hybrid GATv2 branch of the SolubilityAI ensemble, ported verbatim from
solubility_ai_v2.ipynb cells 44/46/48 (graph construction, architecture,
training loop) so the re-established baseline is the same model the project
reported on, only on corrected data.

Unchanged from the notebook: 17-dim atom features, 4-dim bond features, 5 global
descriptors, GATv2Conv(17->32,heads=4) -> GATv2Conv(128->64,heads=2) -> mean pool
-> concat globals -> 64 -> 32 -> 1, AdamW(lr=1e-3, wd=0.01), MSE, 60 epochs,
batch 64, ReduceLROnPlateau(factor=0.5, patience=3).

Changed: torch.manual_seed is joined by numpy/random seeding and
torch.use_deterministic_algorithms is left off (CPU GATv2 scatter ops have no
deterministic implementation); run-to-run variation is therefore reported via
multiple seeds rather than assumed away.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smiles_to_hybrid_graph(smiles, target=0.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        identity = [1 if num == a else 0 for a in [6, 7, 8, 9, 16, 17, 35, 53]]
        hyb = atom.GetHybridization()
        hybridization = [
            1 if hyb == Chem.HybridizationType.SP else 0,
            1 if hyb == Chem.HybridizationType.SP2 else 0,
            1 if hyb == Chem.HybridizationType.SP3 else 0,
        ]
        feat = (identity + hybridization
                + [1 if atom.GetIsAromatic() else 0]
                + [atom.GetTotalNumHs()]
                + [atom.GetFormalCharge()]
                + [1 if atom.HasProp("_ChiralityPossible") else 0]
                + [atom.GetTotalValence()]
                + [atom.GetDegree()])
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[s, e], [e, s]])
        bt = bond.GetBondType()
        bf = [1 if bt == Chem.BondType.SINGLE else 0,
              1 if bt == Chem.BondType.DOUBLE else 0,
              1 if bt == Chem.BondType.TRIPLE else 0,
              1 if bt == Chem.BondType.AROMATIC else 0]
        edge_features.extend([bf, bf])

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    try:
        g = [Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
             Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
    except Exception:
        g = [0.0] * 5

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                global_feats=torch.tensor(g, dtype=torch.float),
                y=torch.tensor([float(target)], dtype=torch.float))


class HybridMolecularGATv2(nn.Module):
    def __init__(self, num_node_features=17, num_edge_features=4, num_global_features=5):
        super().__init__()
        self.gat1 = GATv2Conv(num_node_features, 32, heads=4, edge_dim=num_edge_features)
        self.gat2 = GATv2Conv(128, 64, heads=2, concat=True, edge_dim=num_edge_features)
        self.fc1 = nn.Linear(128 + num_global_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, data):
        x = F.elu(self.gat1(data.x, data.edge_index, data.edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.gat2(x, data.edge_index, data.edge_attr))
        x = global_mean_pool(x, data.batch)
        g = data.global_feats.view(data.num_graphs, -1)
        x = F.relu(self.fc1(torch.cat([x, g], dim=1)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_gatv2(train_graphs, seed, epochs=60):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    model = HybridMolecularGATv2().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    crit = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(batch).squeeze(-1), batch.y)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
        sched.step(total / len(loader.dataset))
    return model


def predict_gatv2(model, graphs):
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            preds.extend(model(batch).squeeze(-1).cpu().numpy().ravel())
    return np.array(preds)


def build_graphs(df, smiles_col="canonical_smiles", target_col="Solubility"):
    """Returns graphs and the positional mask of molecules that built successfully."""
    gs, ok = [], []
    for s, t in zip(df[smiles_col], df[target_col]):
        g = smiles_to_hybrid_graph(s, t)
        ok.append(g is not None)
        if g is not None:
            gs.append(g)
    return gs, np.array(ok)
