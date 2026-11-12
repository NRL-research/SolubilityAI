"""
Molecular featurizers, consolidated from three places the notebook and
dissolvr/ had reinvented independently:
  - notebook's extract_classical_features (Morgan FP + 5 global descriptors)
  - notebook's extract_200_descriptors (full RDKit descriptor list)
  - dissolvr/featurizer.py (full RDKit descriptors + custom "mose" graph
    homomorphism counts + Joback/Abraham thermodynamic proxies)

Exposed as pure functions over a list of SMILES, plus a small registry
(FEATURE_SETS) so the sweep can select a representation by name.
"""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem, Crippen, Descriptors, GraphDescriptors, MACCSkeys, rdFingerprintGenerator,
    rdMolDescriptors, rdMolDescriptors as rdMD, Fragments,
)
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")  # 3D embedding is noisy about failed conformers otherwise


# ---------------------------------------------------------------------------
# Morgan fingerprints
# ---------------------------------------------------------------------------

def morgan_fingerprint(smiles_list, radius=2, n_bits=2048):
    """(n_molecules, n_bits) binary matrix. Uses the modern generator API
    (the notebook used the deprecated AllChem.GetMorganFingerprintAsBitVect)."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        fp = generator.GetFingerprint(mol)
        rows.append(np.array(fp, dtype=np.uint8))
    return np.vstack(rows)


def maccs_keys(smiles_list):
    """(n_molecules, 167) binary matrix -- MACCS structural keys. Only ever
    radius=2/2048-bit Morgan has been tried in the sweep so far; MACCS is a
    completely different (much smaller, hand-curated substructure) fingerprint
    family, cheap to try."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(167, dtype=np.uint8))
            continue
        rows.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8))
    return np.vstack(rows)


def atom_pair_fingerprint(smiles_list, n_bits=2048):
    """(n_molecules, n_bits) -- atom-pair fingerprint: encodes pairs of atoms
    and the topological distance between them. Captures different structural
    information than Morgan's circular-substructure counting."""
    generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        rows.append(np.array(generator.GetFingerprint(mol), dtype=np.uint8))
    return np.vstack(rows)


def topological_torsion_fingerprint(smiles_list, n_bits=2048):
    """(n_molecules, n_bits) -- topological-torsion fingerprint: encodes
    4-atom paths. Complements Morgan (circular) and atom-pair (2-atom +
    distance) with a 3rd distinct structural view."""
    generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        rows.append(np.array(generator.GetFingerprint(mol), dtype=np.uint8))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Global 5-descriptor block (used by the GNN's hybrid head and as a cheap
# classical baseline)
# ---------------------------------------------------------------------------

def global_descriptors_5(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([0.0] * 5)
            continue
        rows.append([
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ])
    return pd.DataFrame(rows, columns=["mol_wt", "log_p", "tpsa", "h_donors", "h_acceptors"])


# ---------------------------------------------------------------------------
# Full RDKit descriptor list (~200 descriptors). dissolvr/featurizer.py
# excludes a correlated subset; exposed as a toggle here instead of two
# separate implementations.
# ---------------------------------------------------------------------------

DISSOLVR_EXCLUDE_PREFIXES = ("BCUT2D", "SMR_VSA", "SlogP_VSA", "VSA_EState", "Chi", "FpDensityMorgan")
DISSOLVR_EXCLUDE_EXACT = {"Ipc", "Kappa3", "MolMR", "HeavyAtomMolWt", "ExactMolWt"}


def rdkit_descriptors(smiles_list, exclude_prefixes=(), exclude_exact=()):
    desc_map = {}
    for name, func in Descriptors.descList:
        if name in exclude_exact:
            continue
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        desc_map[name] = func

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({name: 0.0 for name in desc_map})
            continue
        row = {}
        for name, func in desc_map.items():
            try:
                row[name] = func(mol)
            except Exception:
                row[name] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows, columns=list(desc_map.keys()))
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # Some descriptors (Ipc in particular) can produce finite but astronomically
    # large float64 values (seen up to ~1e158 on large/branched molecules in this
    # dataset) that silently become +-inf on a later cast to float32. Clip to the
    # float32-safe range, matching the notebook's own workaround for this exact
    # issue (cell 35's XGBoost sanitization pass).
    df = df.clip(lower=-3.4e38, upper=3.4e38)
    return df


# ---------------------------------------------------------------------------
# Pillar 1 (functional-group fragment counts) + Pillar 2 (topology indices),
# from src/phase1_featurize.py -- kept for the (currently unbuilt) RAG
# pipeline's retrieval context, and available as a lightweight feature block.
# ---------------------------------------------------------------------------

PILLAR1_FRAGMENTS = {
    "fr_ketone": Fragments.fr_ketone,
    "fr_aldehyde": Fragments.fr_aldehyde,
    "fr_ester": Fragments.fr_ester,
    "fr_ether": Fragments.fr_ether,
    "fr_amide": Fragments.fr_amide,
    "fr_amine": Fragments.fr_NH2,
    "fr_aniline": Fragments.fr_aniline,
    "fr_phenol": Fragments.fr_phenol,
    "fr_alcohol": Fragments.fr_Al_OH,
    "fr_carboxylic_acid": Fragments.fr_COO,
    "fr_halogen": Fragments.fr_halogen,
    "fr_sulfonamide": Fragments.fr_sulfonamd,
    "fr_benzene": Fragments.fr_benzene,
    "fr_pyridine": Fragments.fr_pyridine,
    "fr_nitrile": Fragments.fr_nitrile,
}


def pillar1_pillar2_descriptors(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({})
            continue
        row = {name: func(mol) for name, func in PILLAR1_FRAGMENTS.items()}
        row.update({
            "balaban_j": GraphDescriptors.BalabanJ(mol),
            "bertz_ct": GraphDescriptors.BertzCT(mol),
            "kappa1": GraphDescriptors.Kappa1(mol),
            "kappa2": GraphDescriptors.Kappa2(mol),
            "ring_count": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DISSOLVR's custom "mose" homomorphism counts + Joback/Abraham thermo
# proxies.
# ---------------------------------------------------------------------------

_JOBACK_INCREMENTS = {
    "ch3": ("[CH3;X4;!R]", -5.10), "ch2_c": ("[CH2;X4;!R]", 11.27),
    "ch_c": ("[CH1;X4;!R]", 12.64), "c_c": ("[CH0;X4;!R]", 46.43),
    "ch2_r": ("[CH2;X4;R]", 8.25), "ch_r": ("[CH1;X4;R]", 20.15),
    "c_r": ("[CH0;X4;R]", 37.40), "c=c_c": ("[CX3;!R]=[CX3;!R]", 4.18),
    "c=c_r": ("[c,C;R]=[c,C;R]", 13.02), "F": ("[F]", 9.88),
    "Cl": ("[Cl]", 17.51), "Br": ("[Br]", 26.15), "I": ("[I]", 37.0),
    "oh_a": ("[OH;!#6a]", 20.0), "oh_p": ("[OH;a]", 44.45),
    "ether_c": ("[OD2;!R]([#6])[#6]", 22.42), "ether_r": ("[OD2;R]([#6])[#6]", 31.22),
    "co": ("[CX3]=[OX1]", 26.15), "ester": ("[CX3](=[OX1])[OX2H0]", 30.0),
    "nh2": ("[NH2]", 25.72), "nh_c": ("[NH1;!R]", 27.15), "nh_r": ("[NH1;R]", 30.12),
    "nitro": ("[NX3](=[OX1])=[OX1]", 45.0), "nitrile": ("[NX1]#[CX2]", 33.15),
}
_ATOM_V_MAP = {
    "C": 16.35, "N": 14.39, "O": 12.43, "F": 10.48, "Cl": 20.95, "Br": 26.21,
    "I": 34.53, "S": 22.91, "P": 24.87, "H": 8.71, "Se": 25.10,
    "B": 18.32, "Na": 18.00, "K": 24.00, "Ca": 21.00, "Li": 14.00,
}


def _mose_features(mol):
    A = Chem.GetAdjacencyMatrix(mol).astype(float)
    degrees = np.sum(A, axis=0)
    A2 = np.linalg.matrix_power(A, 2)
    A3, A4, A5, A6, A7, A8 = A2 @ A, None, None, None, None, None
    A4 = A3 @ A
    A5 = A4 @ A
    A6 = A5 @ A
    A7 = A6 @ A
    A8 = A7 @ A
    return {
        "mose_cyc3": float(np.trace(A3)), "mose_cyc4": float(np.trace(A4)),
        "mose_cyc5": float(np.trace(A5)), "mose_cyc6": float(np.trace(A6)),
        "mose_cyc7": float(np.trace(A7)), "mose_cyc8": float(np.trace(A8)),
        "mose_path3": float(np.sum(A2)), "mose_path4": float(np.sum(A3)),
        "mose_path5": float(np.sum(A4)),
        "mose_branched_4": float(np.sum(np.power(degrees, 3))),
        "mose_star_5": float(np.sum(np.power(degrees, 4))),
        "mose_benzene": len(mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1"))),
        "mose_fused": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]"))),
    }


def _thermo_proxies(mol):
    tm_sum = 122.5 + sum(
        len(mol.GetSubstructMatches(Chem.MolFromSmarts(p))) * w for p, w in _JOBACK_INCREMENTS.values()
    )
    pred_tm = max(tm_sum, 150.0)

    rd_hbd, rd_hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    acid_ref = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(s))) for s in ["[OH]c", "C(=O)[OH]"])
    base_ref = sum(
        len(mol.GetSubstructMatches(Chem.MolFromSmarts(s)))
        for s in ["[NH2,NH1,NH0]", "n1ccccc1", "[CX3]=[OX1]"]
    )

    proxy_e = Descriptors.MolMR(mol) / 10.0
    hetero_smarts = "[O,N,S,F,Cl,Br,I,P,Se,B,Na,K,Ca,Li]"
    hetero_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts(hetero_smarts)))
    proxy_s = (hetero_count * 0.2) + (len(mol.GetSubstructMatches(Chem.MolFromSmarts("a1aaaaa1"))) * 0.3)

    v_total = sum(_ATOM_V_MAP.get(a.GetSymbol(), 15.0) for a in mol.GetAtoms())
    v_total += sum(a.GetTotalNumHs() for a in mol.GetAtoms()) * 8.71
    v_total -= 6.56 * mol.GetNumBonds()

    return {
        "pred_Tm": pred_tm,
        "abraham_A": rd_hbd * 0.1 + acid_ref * 0.4,
        "abraham_B": rd_hba * 0.1 + base_ref * 0.3,
        "abraham_S": proxy_s,
        "abraham_E": proxy_e,
        "abraham_V": v_total / 100.0,
    }


def dissolvr_engineered_features(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({})
            continue
        row = {}
        row.update(_mose_features(mol))
        row.update(_thermo_proxies(mol))
        rows.append(row)
    return pd.DataFrame(rows)


def dissolvr_full_features(smiles_list):
    """Matches dissolvr/featurizer.py's MoleculeFeaturizer.transform output."""
    desc = rdkit_descriptors(smiles_list, DISSOLVR_EXCLUDE_PREFIXES, DISSOLVR_EXCLUDE_EXACT)
    engineered = dissolvr_engineered_features(smiles_list)
    df = pd.concat([engineered.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


# ---------------------------------------------------------------------------
# 3D descriptors: WHIM / RDF / 3D-autocorrelation. Requires embedding a 3D
# conformer per molecule (ETKDG) -- genuinely new information not captured
# by any 2D descriptor set used so far, since solubility relates to 3D
# molecular shape and surface area. Embedding can fail to converge for some
# molecules (especially large/flexible ones); those get an all-zero row
# rather than crashing the batch.
#
# GETAWAY deliberately excluded: benchmarked at 7.8s for a 7-atom molecule
# and 48s for an 18-atom one (RDKit's implementation scales very badly with
# atom count -- looks like an O(n^3)+ leverage-matrix computation). This
# dataset has molecules up to 388 atoms; projected full-dataset cost with
# GETAWAY included was ~76 hours (measured: 2742s for just 100 molecules).
# WHIM/RDF/AUTOCORR3D are near-instant by comparison.
# ---------------------------------------------------------------------------

def _embed_3d(mol):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol


def descriptors_3d(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        row = {}
        mol3d = _embed_3d(mol) if mol is not None else None
        if mol3d is not None:
            try:
                AllChem.ComputeGasteigerCharges(mol3d)
                row.update({f"whim_{i}": v for i, v in enumerate(rdMD.CalcWHIM(mol3d))})
            except Exception:
                pass
            try:
                row.update({f"rdf_{i}": v for i, v in enumerate(rdMD.CalcRDF(mol3d))})
            except Exception:
                pass
            try:
                row.update({f"autocorr3d_{i}": v for i, v in enumerate(rdMD.CalcAUTOCORR3D(mol3d))})
            except Exception:
                pass
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df = df.clip(lower=-3.4e38, upper=3.4e38)
    return df


# ---------------------------------------------------------------------------
# Mordred descriptor library (~1800 descriptors, 2D subset used here for
# speed/robustness -- 3D Mordred descriptors would need the same conformer
# embedding as descriptors_3d() above and are much slower for marginal extra
# coverage given descriptors_3d already covers the 3D angle).
# ---------------------------------------------------------------------------

def mordred_descriptors(smiles_list):
    from mordred import Calculator, descriptors as mordred_descriptor_list

    calc = Calculator(mordred_descriptor_list, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_idx]

    result_df = calc.pandas(valid_mols, quiet=True)
    # Mordred returns per-cell error sentinels (mordred.error.Missing/Error
    # instances) for descriptors it couldn't compute on a given molecule,
    # mixed with real bool/int/float values across columns. Coerce
    # cell-by-cell rather than relying on column dtype inference: that
    # chokes on the sentinel objects, and separately on newer pandas'
    # stricter bool->float64 casting when assigning into a pre-typed frame.
    numeric = result_df.map(
        lambda v: float(v) if isinstance(v, (int, float, bool)) and not isinstance(v, str) else np.nan
    )
    numeric.index = valid_idx

    full = numeric.reindex(range(len(smiles_list)))
    full = full.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    full = full.clip(lower=-3.4e38, upper=3.4e38)
    return full.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Frozen ChemBERTa embeddings: a pretrained chemical language model used as
# a fixed feature extractor (no fine-tuning), fed into the tree models. The
# notebook attempted a full ChemBERTa fine-tune but it was cancelled
# mid-run; this is a much cheaper way to get some of the same pretrained
# signal into the sweep.
# ---------------------------------------------------------------------------

_CHEMBERTA_CACHE = {}


def chemberta_embeddings(smiles_list, model_name="DeepChem/ChemBERTa-77M-MTR", batch_size=64):
    import torch as _torch
    from transformers import AutoTokenizer, AutoModel

    if model_name not in _CHEMBERTA_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model.to(device).eval()
        _CHEMBERTA_CACHE[model_name] = (tokenizer, model, device)
    tokenizer, model, device = _CHEMBERTA_CACHE[model_name]

    all_embeddings = []
    with _torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # mean-pool token embeddings over the attention mask, standard
            # sentence-embedding approach for encoder-only transformers.
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature selection: mutual-information ranking on top of a large feature
# matrix, as its own selectable representation (fit on train only, applied
# to val/test with the same selected columns -- callers must pass the same
# `k` and refit only on train to avoid leaking val/test into selection).
# ---------------------------------------------------------------------------

def select_top_k_features(X_train, y_train, X_other_list, k=100, seed=42):
    """Returns (X_train_selected, [X_other_selected, ...]) using mutual
    information computed on (X_train, y_train) only."""
    from sklearn.feature_selection import mutual_info_regression

    k = min(k, X_train.shape[1])
    scores = mutual_info_regression(X_train, y_train, random_state=seed)
    top_idx = np.argsort(scores)[::-1][:k]
    selected_train = X_train[:, top_idx]
    selected_others = [X[:, top_idx] for X in X_other_list]
    return selected_train, selected_others, top_idx


# ---------------------------------------------------------------------------
# Graph builder for the GNN models (GCN / GATv2 / Hybrid GATv2), matching the
# notebook's smiles_to_hybrid_graph.
# ---------------------------------------------------------------------------

_ATOMIC_NUMS = [6, 7, 8, 9, 16, 17, 35, 53]


def smiles_to_graph(smiles, target=None, include_global=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        identity = [1 if num == a else 0 for a in _ATOMIC_NUMS]
        hyb = atom.GetHybridization()
        hybridization = [
            1 if hyb == Chem.HybridizationType.SP else 0,
            1 if hyb == Chem.HybridizationType.SP2 else 0,
            1 if hyb == Chem.HybridizationType.SP3 else 0,
        ]
        feat = (
            identity + hybridization
            + [1 if atom.GetIsAromatic() else 0]
            + [atom.GetTotalNumHs()]
            + [atom.GetFormalCharge()]
            + [1 if atom.HasProp("_ChiralityPossible") else 0]
            + [atom.GetTotalValence()]
            + [atom.GetDegree()]
        )
        atom_features.append(feat)
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([[start, end], [end, start]])
        b_type = bond.GetBondType()
        b_feat = [
            1 if b_type == Chem.BondType.SINGLE else 0,
            1 if b_type == Chem.BondType.DOUBLE else 0,
            1 if b_type == Chem.BondType.TRIPLE else 0,
            1 if b_type == Chem.BondType.AROMATIC else 0,
        ]
        edge_features.extend([b_feat, b_feat])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    kwargs = dict(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if include_global:
        g = global_descriptors_5([smiles]).iloc[0]
        kwargs["global_feats"] = torch.tensor(g.values.astype(np.float32), dtype=torch.float)
    if target is not None:
        kwargs["y"] = torch.tensor([target], dtype=torch.float)
    return Data(**kwargs)


def smiles_list_to_graphs(smiles_list, targets=None, include_global=True):
    if targets is None:
        targets = [None] * len(smiles_list)
    graphs = [smiles_to_graph(s, t, include_global) for s, t in zip(smiles_list, targets)]
    return [g for g in graphs if g is not None]


# ---------------------------------------------------------------------------
# Registry: named feature sets the sweep can select between.
# ---------------------------------------------------------------------------

def build_feature_matrix(smiles_list, feature_set):
    if feature_set == "morgan":
        return morgan_fingerprint(smiles_list).astype(np.float32)
    if feature_set == "morgan+global5":
        fp = morgan_fingerprint(smiles_list).astype(np.float32)
        g5 = global_descriptors_5(smiles_list).values.astype(np.float32)
        return np.hstack([fp, g5])
    if feature_set == "rdkit_descriptors":
        return rdkit_descriptors(smiles_list).values.astype(np.float32)
    if feature_set == "dissolvr_full":
        return dissolvr_full_features(smiles_list).values.astype(np.float32)
    if feature_set == "rdkit_descriptors+morgan":
        desc = rdkit_descriptors(smiles_list).values.astype(np.float32)
        fp = morgan_fingerprint(smiles_list).astype(np.float32)
        return np.hstack([desc, fp])
    if feature_set == "maccs":
        return maccs_keys(smiles_list).astype(np.float32)
    if feature_set == "morgan_r3_4096":
        return morgan_fingerprint(smiles_list, radius=3, n_bits=4096).astype(np.float32)
    if feature_set == "atom_pair":
        return atom_pair_fingerprint(smiles_list).astype(np.float32)
    if feature_set == "topological_torsion":
        return topological_torsion_fingerprint(smiles_list).astype(np.float32)
    if feature_set == "fingerprints_combined":
        # Morgan + MACCS + atom-pair + torsion: three structurally distinct
        # fingerprint families concatenated, never tried together.
        morgan = morgan_fingerprint(smiles_list).astype(np.float32)
        maccs = maccs_keys(smiles_list).astype(np.float32)
        ap = atom_pair_fingerprint(smiles_list).astype(np.float32)
        tt = topological_torsion_fingerprint(smiles_list).astype(np.float32)
        return np.hstack([morgan, maccs, ap, tt])
    if feature_set == "rdkit_full+engineered":
        # Full (unexcluded) RDKit descriptors + DISSOLVR's mose/thermo
        # engineered features, combined -- neither existing representation
        # includes both; dissolvr_full excludes several RDKit descriptors
        # DISSOLVR's authors dropped for correlation reasons that may still
        # carry signal for a different model.
        desc = rdkit_descriptors(smiles_list).reset_index(drop=True)
        eng = dissolvr_engineered_features(smiles_list).reset_index(drop=True)
        combined = pd.concat([desc, eng], axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return combined.values.astype(np.float32)
    if feature_set == "descriptors_3d":
        return descriptors_3d(smiles_list).values.astype(np.float32)
    if feature_set == "rdkit_descriptors+3d":
        desc = rdkit_descriptors(smiles_list).reset_index(drop=True)
        d3 = descriptors_3d(smiles_list).reset_index(drop=True)
        combined = pd.concat([desc, d3], axis=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return combined.values.astype(np.float32)
    if feature_set == "mordred":
        return mordred_descriptors(smiles_list).values.astype(np.float32)
    if feature_set == "chemberta":
        return chemberta_embeddings(smiles_list)
    if feature_set == "chemberta+rdkit_descriptors":
        emb = chemberta_embeddings(smiles_list)
        desc = rdkit_descriptors(smiles_list).values.astype(np.float32)
        return np.hstack([emb, desc])
    raise ValueError(f"Unknown feature_set: {feature_set}")


# Original 5, kept exactly as before (still the primary sweep space).
FEATURE_SETS = [
    "morgan",
    "morgan+global5",
    "rdkit_descriptors",
    "dissolvr_full",
    "rdkit_descriptors+morgan",
]

# New representations added for the "beat DISSOLVR" experiment round. Kept
# as a separate list rather than merged into FEATURE_SETS so existing sweep
# results/resume logic (keyed by feature_set name against cached .npy files)
# isn't disturbed -- callers that want the expanded space use ALL_FEATURE_SETS.
EXPANDED_FEATURE_SETS = [
    "maccs",
    "morgan_r3_4096",
    "atom_pair",
    "topological_torsion",
    "fingerprints_combined",
    "rdkit_full+engineered",
    "descriptors_3d",
    "rdkit_descriptors+3d",
    "mordred",
    "chemberta",
    "chemberta+rdkit_descriptors",
]

ALL_FEATURE_SETS = FEATURE_SETS + EXPANDED_FEATURE_SETS
