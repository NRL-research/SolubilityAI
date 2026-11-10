"""Pure-PyTorch message-passing neural network (edge-conditioned, residual), no external
graph library.  Copied verbatim into each saved GNN model folder.

Graph input comes from featurize.graphs(smiles) -> list of (x[n,Fa], edge_index[2,e], edge_attr[e,Fb]).
Optional 'global' input: preprocessed RDKit descriptor vector concatenated at readout (hybrid).
"""
import numpy as np
import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, n_atom, n_bond, hidden=200, layers=4, n_global=0, dropout=0.1, head=256):
        super().__init__()
        self.atom = nn.Linear(n_atom, hidden)
        self.msg = nn.ModuleList([nn.Linear(hidden + n_bond, hidden) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                                  for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.n_global = n_global
        self.head = nn.Sequential(nn.Linear(2 * hidden + n_global, head), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(head, head // 2), nn.ReLU(), nn.Linear(head // 2, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ei, ea, batch, n_graphs, g=None):
        h = torch.relu(self.atom(x))
        src, dst = ei
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = torch.relu(msg(torch.cat([h[src], ea], 1)))
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = norm(h + self.drop(upd(torch.cat([h, agg], 1))))
        s = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)
        cnt = torch.zeros(n_graphs).index_add_(0, batch, torch.ones(len(batch))).clamp(min=1).unsqueeze(1)
        r = torch.cat([s / cnt, s / 10.0], 1)  # mean + scaled sum pooling
        if self.n_global:
            r = torch.cat([r, g], 1)
        return self.head(r).squeeze(1)


def collate(graphs, idx, gx=None):
    xs, eis, eas, bs = [], [], [], []
    off = 0
    for k, i in enumerate(idx):
        x, ei, ea = graphs[i]
        xs.append(x); eis.append(ei + off); eas.append(ea); bs.append(np.full(len(x), k))
        off += len(x)
    out = [torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(eis, 1)),
           torch.from_numpy(np.concatenate(eas)), torch.from_numpy(np.concatenate(bs)), len(idx)]
    out.append(torch.from_numpy(gx[idx]) if gx is not None else None)
    return out


@torch.no_grad()
def predict_graphs(net, graphs, gx=None, bs=256):
    """Returns predictions in the STANDARDIZED target scale (caller un-standardizes)."""
    net.eval()
    out = []
    for s in range(0, len(graphs), bs):
        idx = np.arange(s, min(s + bs, len(graphs)))
        x, ei, ea, b, n, g = collate(graphs, idx, gx)
        out.append(net(x, ei, ea, b, n, g).numpy())
    return np.concatenate(out).astype(float)


def train(graphs, y_std, gx=None, seed=0, epochs=100, lr=1e-3, bs=64, wd=1e-5, arch=None, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = MPNN(**arch)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(graphs) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.RandomState(seed)
    yt = torch.tensor(y_std, dtype=torch.float32)
    for ep in range(epochs):
        net.train(); perm = rng.permutation(len(graphs)); tot = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            x, ei, ea, b, n, g = collate(graphs, idx, gx)
            loss = torch.mean((net(x, ei, ea, b, n, g) - yt[idx]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        if log and (ep % 10 == 0 or ep == epochs - 1):
            log(f"epoch {ep} train_mse_std {tot / len(perm):.4f}")
    return net
