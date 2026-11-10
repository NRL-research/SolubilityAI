"""Step 02: similarity-cluster group split (train/test) + grouped CV folds, with leakage diagnostics.

Method
  * Group key = connected components of (Butina cluster on parent Morgan FP, Tanimoto
    distance cutoff 0.6) UNION (same parent InChIKey connectivity block).  Salt forms,
    stereoisomers and close analogues of a parent therefore always share a group.
  * Butina treats acyclic molecules exactly like ring-containing ones (no empty-scaffold
    mega-group), which is the failure mode of Bemis-Murcko scaffold splits.
  * Groups are shuffled and assigned to test until >= 20% of molecules; 200 shuffles (seeds 42..241)
    are scored on train/test balance (LogS KS stat + acyclic-fraction gap + size gap) and the best kept.
  * CV: StratifiedGroupKFold(5, shuffle, seed 42) on the training set, stratified on LogS deciles.
Diagnostics compare against (a) random split and (b) Murcko scaffold split.
Outputs: data/05_split_assignments.csv, data/06_train.csv, data/06_test.csv, results/02_split_diagnostics.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import ks_2samp

SEED = 42
CUTOFF = 0.6
TEST_FRAC = 0.20
R = Path(__file__).resolve().parents[1]
df = pd.read_csv(R / "data/04_aqsoldb_curated.csv")
n = len(df)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mols = [Chem.MolFromSmiles(s) for s in df.parent_smiles]
fps = [gen.GetFingerprint(m) for m in mols]

# ---- Butina clustering (full lower-triangle distance list) ----
dists = []
for i in range(1, n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1 - s for s in sims])
clusters = Butina.ClusterData(dists, n, CUTOFF, isDistData=True, reordering=True)
del dists
clus = np.empty(n, int)
for ci, members in enumerate(clusters):
    clus[list(members)] = ci

# ---- union-find with parent identity ----
par = list(range(n))
def find(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: par[rb] = ra
for key in [clus, df.parent_block1.values]:
    first = {}
    for i, k in enumerate(key):
        if k in first: union(first[k], i)
        else: first[k] = i
roots = np.array([find(i) for i in range(n)])
_, group = np.unique(roots, return_inverse=True)
df["butina_cluster"] = clus
df["group"] = group

# ---- test assignment ----
# Try SPLIT_SEEDS candidate group shuffles; keep the one whose test set best matches train on
# (LogS distribution KS statistic) + (acyclic fraction) + (test size).  Only labels/structure
# balance is used -- no model is ever trained for this choice.
sizes = pd.Series(group).value_counts()
acyc_all = (~df.has_ring_parent).values
def make_split(seed):
    rng = np.random.RandomState(seed)
    test_groups, tot = set(), 0
    for g in rng.permutation(np.unique(group)):
        if tot >= TEST_FRAC * n: break
        test_groups.add(g); tot += sizes[g]
    return df.group.isin(test_groups).values
best = None
for sd in range(200):
    t = make_split(SEED + sd)
    score = (ks_2samp(df.LogS[t], df.LogS[~t]).statistic + abs(acyc_all[t].mean() - acyc_all[~t].mean())
             + abs(t.mean() - TEST_FRAC))
    if best is None or score < best[0]:
        best = (score, SEED + sd, t)
SPLIT_SEED = best[1]
df["split"] = np.where(best[2], "test", "train")
# ---- CV folds on train ----
tr = df[df.split == "train"]
ybin = pd.qcut(tr.LogS, 10, labels=False, duplicates="drop")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
df["cv_fold"] = -1
for k, (_, va) in enumerate(sgkf.split(tr, ybin, tr.group)):
    df.loc[tr.index[va], "cv_fold"] = k

# ---- diagnostics ----
def nn_sim(test_idx, train_idx):
    tfp = [fps[i] for i in train_idx]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], tfp)) for i in test_idx])

def summarize(name, is_test):
    te, trn = np.where(is_test)[0], np.where(~is_test)[0]
    s = nn_sim(te, trn)
    acyc = ~df.has_ring_parent.values
    shared_parent = len(set(df.parent_block1[is_test]) & set(df.parent_block1[~is_test]))
    return dict(method=name, n_train=int(len(trn)), n_test=int(len(te)),
                acyclic_frac_train=float(acyc[trn].mean()), acyclic_frac_test=float(acyc[te].mean()),
                nn_sim_median=float(np.median(s)), frac_test_nn_ge_0p8=float((s >= 0.8).mean()),
                frac_test_nn_eq_1=float((s >= 0.999).mean()), parent_blocks_shared=shared_parent,
                logS_ks_p=float(ks_2samp(df.LogS[is_test], df.LogS[~is_test]).pvalue),
                logS_mean_train=float(df.LogS[~is_test].mean()), logS_mean_test=float(df.LogS[is_test].mean()))

diag = {"cluster_params": dict(fp="Morgan r2 2048 bits on parent", butina_distance_cutoff=CUTOFF, seed=SEED, chosen_split_seed=int(SPLIT_SEED), split_seed_candidates="42..241", cv_seed=SEED),
        "n_butina_clusters": int(clus.max() + 1), "n_groups": int(group.max() + 1),
        "largest_group": int(sizes.max()), "groups_size1": int((sizes == 1).sum())}
diag["ours"] = summarize("butina_group", (df.split == "test").values)
rnd = np.random.RandomState(SEED).rand(n) < TEST_FRAC
diag["random_baseline"] = summarize("random", rnd)
# Murcko scaffold split, the conventional way (all acyclic molecules share scaffold '')
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
df["murcko"] = scaf
ss = pd.Series(scaf).value_counts()
diag["murcko_empty_scaffold_group_size"] = int(ss.get("", 0))
order = sorted(ss.index, key=lambda s: -ss[s])  # deterministic chemprop-style: big scaffolds to train
cum, tscaf = 0, set()
for s in order[::-1]:
    if cum >= TEST_FRAC * n: break
    tscaf.add(s); cum += ss[s]
diag["murcko_baseline"] = summarize("murcko_scaffold_largest_to_train", pd.Series(scaf).isin(tscaf).values)

# CV fold diagnostics
folds = []
for k in range(5):
    va = df[df.cv_fold == k]; trk = df[(df.split == "train") & (df.cv_fold != k)]
    folds.append(dict(fold=k, n_val=len(va), acyclic_frac_val=float((~va.has_ring_parent).mean()),
                      logS_mean_val=float(va.LogS.mean()),
                      groups_shared_with_train=len(set(va.group) & set(trk.group)),
                      parent_blocks_shared=len(set(va.parent_block1) & set(trk.parent_block1))))
diag["cv_folds"] = folds
json.dump(diag, open(R / "results/02_split_diagnostics.json", "w"), indent=2)
df.drop(columns=["murcko"]).to_csv(R / "data/05_split_assignments.csv", index=False)
df[df.split == "train"].drop(columns=["murcko"]).to_csv(R / "data/06_train.csv", index=False)
df[df.split == "test"].drop(columns=["murcko"]).to_csv(R / "data/06_test.csv", index=False)
print(json.dumps(diag, indent=2))
