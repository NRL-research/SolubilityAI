"""
Model builders extracted from the notebook's GCN / GATv2 / Hybrid GATv2 /
XGBoost / CatBoost cells, plus LightGBM (listed in requirements.txt but never
actually used anywhere in the original codebase).

Key methodological fix versus the notebook: train_gnn() early-stops on a
validation loader, not on training loss (the notebook trained GNNs for a
fixed epoch count with no held-out signal at all).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tree models
# ---------------------------------------------------------------------------

def build_tree_model(model_type, params, seed=42):
    params = dict(params)
    if model_type == "xgboost":
        import xgboost as xgb
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("n_jobs", -1)
        params["random_state"] = seed
        return xgb.XGBRegressor(**params)
    if model_type == "catboost":
        from catboost import CatBoostRegressor
        params.setdefault("loss_function", "RMSE")
        params.setdefault("verbose", 0)
        params.setdefault("allow_writing_files", False)
        params["random_seed"] = seed
        return CatBoostRegressor(**params)
    if model_type == "lightgbm":
        import lightgbm as lgb
        params.setdefault("objective", "regression")
        params.setdefault("n_jobs", -1)
        params.setdefault("verbose", -1)
        params["random_state"] = seed
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"Unknown model_type: {model_type}")


DEFAULT_TREE_SEARCH_SPACE = {
    # Widened from the first sweep round (which plateaued at RMSE=0.8137 --
    # worth checking whether that's a real ceiling or just this space's old
    # boundary): added min_child_weight/gamma/reg_alpha/reg_lambda (XGBoost),
    # bagging_temperature/random_strength (CatBoost), min_child_samples/
    # reg_alpha/reg_lambda (LightGBM).
    "xgboost": {
        "n_estimators": (200, 1200),
        "learning_rate": (0.005, 0.2, "log"),
        "max_depth": (3, 12),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 20),
        "gamma": (1e-8, 5.0, "log"),
        "reg_alpha": (1e-8, 10.0, "log"),
        "reg_lambda": (1e-8, 10.0, "log"),
    },
    "catboost": {
        "iterations": (300, 1200),
        "learning_rate": (0.005, 0.2, "log"),
        "depth": (4, 10),
        "l2_leaf_reg": (1.0, 15.0),
        "bagging_temperature": (0.0, 5.0),
        "random_strength": (1e-8, 5.0, "log"),
    },
    "lightgbm": {
        "n_estimators": (200, 1200),
        "learning_rate": (0.005, 0.2, "log"),
        "num_leaves": (16, 400),
        "max_depth": (3, 14),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_samples": (5, 100),
        "reg_alpha": (1e-8, 10.0, "log"),
        "reg_lambda": (1e-8, 10.0, "log"),
    },
}


# ---------------------------------------------------------------------------
# Classic sklearn model families -- none of these were tried in the first
# sweep round at all. SVR/GaussianProcess are standard QSAR baselines that
# are often competitive on datasets this size; RandomForest/ExtraTrees have
# a different bias/variance profile than gradient boosting; HistGB/MLP are
# fast, zero-extra-dependency options worth checking.
# ---------------------------------------------------------------------------

def build_sklearn_model(model_type, params, seed=42):
    params = dict(params)
    if model_type == "svr":
        from sklearn.svm import SVR
        return SVR(**params)
    if model_type == "gaussian_process":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        length_scale = params.pop("length_scale", 1.0)
        alpha = params.pop("alpha", 1e-6)
        noise_level = params.pop("noise_level", 1e-3)
        kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        return GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=seed, **params)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        params["random_state"] = seed
        params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**params)
    if model_type == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        params["random_state"] = seed
        params.setdefault("n_jobs", -1)
        return ExtraTreesRegressor(**params)
    if model_type == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        params["random_state"] = seed
        return HistGradientBoostingRegressor(**params)
    if model_type == "mlp":
        from sklearn.neural_network import MLPRegressor
        if isinstance(params.get("hidden_layer_sizes"), str):
            params["hidden_layer_sizes"] = MLP_HIDDEN_LAYER_SIZES_MAP[params["hidden_layer_sizes"]]
        params["random_state"] = seed
        params.setdefault("early_stopping", True)
        params.setdefault("max_iter", 2000)
        return MLPRegressor(**params)
    raise ValueError(f"Unknown sklearn model_type: {model_type}")


DEFAULT_SKLEARN_SEARCH_SPACE = {
    "svr": {
        "C": (1e-2, 1e3, "log"),
        "epsilon": (1e-4, 1.0, "log"),
        "gamma": (1e-5, 1e1, "log"),
    },
    "gaussian_process": {
        "length_scale": (0.1, 50.0, "log"),
        "alpha": (1e-8, 1e-1, "log"),
        "noise_level": (1e-6, 1.0, "log"),
    },
    "random_forest": {
        # n_estimators capped much lower than the gradient-boosted trees:
        # sklearn's RandomForest/ExtraTrees don't have XGBoost/LightGBM's
        # histogram-binned splitting, so cost per tree grows much faster
        # with feature-matrix width. Measured: one trial at n_estimators=566
        # on a 2265-dim feature set took ~58 minutes. See
        # RANDOM_FOREST_FAMILY_FEATURE_SETS below for the matching
        # feature-set restriction (same fix pattern as the SVR/GP kernel
        # methods).
        "n_estimators": (100, 400),
        "max_depth": (4, 30),
        "min_samples_leaf": (1, 20),
        "max_features": (0.3, 1.0),
    },
    "extra_trees": {
        "n_estimators": (100, 400),
        "max_depth": (4, 30),
        "min_samples_leaf": (1, 20),
        "max_features": (0.3, 1.0),
    },
    "hist_gradient_boosting": {
        "max_iter": (100, 1000),
        "learning_rate": (0.01, 0.3, "log"),
        "max_depth": (3, 16),
        "max_leaf_nodes": (15, 255),
        "l2_regularization": (1e-8, 10.0, "log"),
    },
    "mlp": {
        # String labels, not tuples: Optuna's SQLite storage round-trips
        # categorical choices through JSON, which turns a tuple like
        # (256, 128) into a list [256, 128] -- on the next trial, TPE's
        # sampler compares that JSON-deserialized list against the original
        # tuple choices and `[256, 128] in ((256, 128), ...)` is False (a
        # list is never equal to a tuple in Python), raising
        # "not in {choices}" and crashing the whole run. Confirmed: this
        # killed the sweep after 15 trials. MLP_HIDDEN_LAYER_SIZES_MAP below
        # converts the string back to a real tuple at model-build time.
        "hidden_layer_sizes": ["128", "256", "128_64", "256_128", "256_128_64"],
        "alpha": (1e-6, 1e-1, "log"),
        "learning_rate_init": (1e-4, 1e-2, "log"),
    },
}

MLP_HIDDEN_LAYER_SIZES_MAP = {
    "128": (128,), "256": (256,), "128_64": (128, 64),
    "256_128": (256, 128), "256_128_64": (256, 128, 64),
}

# SVR and GaussianProcess don't scale well to high-dimensional sparse
# fingerprints (thousands of bits) or to n_samples > a few thousand for GP
# specifically (O(n^3) kernel inversion) -- restrict their sweep candidates
# to the dense, moderate-width descriptor feature sets.
SKLEARN_KERNEL_METHOD_FEATURE_SETS = [
    "rdkit_descriptors", "dissolvr_full", "rdkit_full+engineered",
    "descriptors_3d", "rdkit_descriptors+3d", "chemberta", "morgan+global5",
]

# RandomForest/ExtraTrees: excludes the widest feature sets (fingerprints_
# combined at 6311 dims, morgan_r3_4096, rdkit_descriptors+morgan at 2265,
# mordred at 1613) -- see the n_estimators comment above for the measured
# cost. Moderate-width sets (chemberta+rdkit_descriptors at 601,
# rdkit_full+engineered at 236, plain fingerprints at 2048) stay in.
RANDOM_FOREST_FAMILY_FEATURE_SETS = [
    "morgan", "morgan+global5", "rdkit_descriptors", "dissolvr_full",
    "maccs", "atom_pair", "topological_torsion", "rdkit_full+engineered",
    "descriptors_3d", "rdkit_descriptors+3d", "chemberta", "chemberta+rdkit_descriptors",
]


# ---------------------------------------------------------------------------
# GNN architectures (ported from the notebook)
# ---------------------------------------------------------------------------

class MolecularGCN(nn.Module):
    def __init__(self, num_node_features, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden * 2)
        self.conv3 = GCNConv(hidden * 2, hidden)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc2(x)


class MolecularGATv2(nn.Module):
    def __init__(self, num_node_features, num_edge_features, heads=4):
        super().__init__()
        self.gat1 = GATv2Conv(num_node_features, 32, heads=heads, edge_dim=num_edge_features)
        self.gat2 = GATv2Conv(32 * heads, 64, heads=2, concat=True, edge_dim=num_edge_features)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class HybridMolecularGATv2(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features=5, heads=4):
        super().__init__()
        self.gat1 = GATv2Conv(num_node_features, 32, heads=heads, edge_dim=num_edge_features)
        self.gat2 = GATv2Conv(32 * heads, 64, heads=2, concat=True, edge_dim=num_edge_features)
        self.fc1 = nn.Linear(128 + num_global_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        num_graphs = data.num_graphs
        global_vector = data.global_feats.view(num_graphs, -1)
        combined = torch.cat([x, global_vector], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


GNN_BUILDERS = {
    "gcn": lambda: MolecularGCN(num_node_features=17),
    "gatv2": lambda: MolecularGATv2(num_node_features=17, num_edge_features=4),
    "hybrid_gatv2": lambda: HybridMolecularGATv2(num_node_features=17, num_edge_features=4, num_global_features=5),
}


def train_gnn(model, train_graphs, val_graphs, epochs=100, lr=1e-3, weight_decay=0.01,
              batch_size=64, patience=15, device=None, seed=42, verbose=False):
    """Trains with early stopping on val RMSE (the notebook never used a
    validation set for this at all). Returns the best-val-epoch model."""
    torch.manual_seed(seed)
    device = device or get_device()
    model = model.to(device)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_since_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out.squeeze(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_mse = total_loss / len(train_loader.dataset)

        val_preds, val_targets = predict_gnn(model, val_loader, device, return_targets=True)
        val_rmse = float(np.sqrt(np.mean((val_preds - val_targets) ** 2)))
        scheduler.step(val_rmse)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_rmse": val_rmse})

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:03d}  train_mse={train_mse:.4f}  val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if patience is not None and epochs_since_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_rmse


def predict_gnn(model, loader_or_graphs, device=None, batch_size=64, return_targets=False):
    device = device or get_device()
    model = model.to(device)
    model.eval()
    if isinstance(loader_or_graphs, list):
        loader = DataLoader(loader_or_graphs, batch_size=batch_size, shuffle=False)
    else:
        loader = loader_or_graphs

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            if out.dim() > 1:
                out = out.squeeze(-1)
            preds.extend(out.cpu().numpy().flatten())
            if hasattr(batch, "y") and batch.y is not None:
                targets.extend(batch.y.cpu().numpy().flatten())

    preds = np.array(preds)
    if return_targets:
        return preds, np.array(targets)
    return preds
