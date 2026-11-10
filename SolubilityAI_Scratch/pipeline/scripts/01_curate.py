"""Step 01: parse, standardize, decontaminate vs SC1/SC2, curate AqSolDB.

Inputs : data/00_raw/*.csv (unchanged copies of provided files)
Outputs: data/01_aqsoldb_parsed.csv
         data/02_sc_reference_keys.csv
         data/03_removed_sc_overlap.csv, data/03_aqsoldb_decontaminated.csv
         data/04_removed_curation.csv,  data/04_aqsoldb_curated.csv
         results/01_curation_log.json
Deterministic (no randomness).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chem import *  # noqa

R = Path(__file__).resolve().parents[1]
D = R / "data"
log = {}

# ---------- AqSolDB ----------
a = pd.read_csv(D / "00_raw/AqSolDB.csv")
log["aqsoldb_raw_rows"] = len(a)
rows = []
for r in a.itertuples():
    m = mol_from_smiles(r.SMILES)
    fm = standardize_full(m)
    p = parent(m)
    rows.append(dict(
        parsed=m is not None,
        std_smiles=Chem.MolToSmiles(fm) if fm is not None else None,
        std_nonstereo=nonstereo_smiles(fm),
        ik_full_recomputed=inchikey(m),
        parent_smiles=Chem.MolToSmiles(p) if p is not None else None,
        parent_ik=inchikey(p),
        n_components=len(r.SMILES.split(".")),
        has_carbon=(m is not None and any(at.GetAtomicNum() == 6 for at in m.GetAtoms())),
        n_heavy=(m.GetNumHeavyAtoms() if m is not None else 0),
        has_ring_parent=(p is not None and p.GetRingInfo().NumRings() > 0),
    ))
a = pd.concat([a, pd.DataFrame(rows)], axis=1)
a["ik_block1_given"] = a.InChIKey.map(ik_block1)
a["ik_block1_recomp"] = a.ik_full_recomputed.map(ik_block1)
a["parent_block1"] = a.parent_ik.map(ik_block1)
a["name_norm"] = a.Name.map(norm_name)
a.to_csv(D / "01_aqsoldb_parsed.csv", index=False)
log["unparseable"] = int((~a.parsed).sum())

# ---------- SC reference keys ----------
sc_frames = []
for f, set_name in [("SC1_processed", "SC1"), ("SC2_tight_processed", "SC2_tight"), ("SC2_loose_processed", "SC2_loose")]:
    s = pd.read_csv(D / f"00_raw/{f}.csv")
    for r in s.itertuples():
        keys = set()
        for smi in {r.smiles, r.canonical_smiles}:
            m = mol_from_smiles(smi)
            if m is None:
                continue
            p = parent(m)
            keys |= {("full", inchikey(m)), ("block1", ik_block1(inchikey(m))),
                     ("parent_block1", ik_block1(inchikey(p)))}
        keys |= {("full", r.inchikey), ("block1", ik_block1(r.inchikey))}
        sc_frames.append(dict(sc_set=set_name, name=r.name, canonical_smiles=r.canonical_smiles,
                              given_inchikey=r.inchikey, name_norm=norm_name(r.name),
                              keys=";".join(sorted(f"{t}:{k}" for t, k in keys if k))))
sc = pd.DataFrame(sc_frames)
sc.to_csv(D / "02_sc_reference_keys.csv", index=False)

sc_block1 = set()
for ks in sc["keys"]:
    for k in ks.split(";"):
        t, v = k.split(":", 1)
        sc_block1.add(v.split("-")[0])  # every key reduced to connectivity block
sc_full = {k.split(":", 1)[1] for ks in sc["keys"] for k in ks.split(";") if k.startswith("full:")}
sc_names = set(sc.name_norm.dropna())
log["sc_compounds"] = int(len(sc)); log["sc_unique_block1"] = len(sc_block1)

# ---------- decontamination (union of criteria, deliberately conservative) ----------
a["hit_full"] = a.InChIKey.isin(sc_full) | a.ik_full_recomputed.isin(sc_full)
a["hit_block1"] = a.ik_block1_given.isin(sc_block1) | a.ik_block1_recomp.isin(sc_block1)
a["hit_parent"] = a.parent_block1.isin(sc_block1)
a["hit_name"] = a.name_norm.isin(sc_names)
a["sc_overlap"] = a.hit_full | a.hit_block1 | a.hit_parent | a.hit_name
for c in ["hit_full", "hit_block1", "hit_parent", "hit_name", "sc_overlap"]:
    log[f"decon_{c}"] = int(a[c].sum())
log["decon_only_by_name"] = int((a.hit_name & ~(a.hit_full | a.hit_block1 | a.hit_parent)).sum())
log["decon_only_by_parent_or_block1"] = int(((a.hit_block1 | a.hit_parent) & ~a.hit_full).sum())
a[a.sc_overlap].to_csv(D / "03_removed_sc_overlap.csv", index=False)
dec = a[~a.sc_overlap].copy()
dec.to_csv(D / "03_aqsoldb_decontaminated.csv", index=False)
log["after_decontamination"] = len(dec)

# ---------- further curation ----------
removed = []
bad = ~dec.parsed | dec.std_smiles.isna() | dec.parent_ik.isna()
removed.append(dec[bad].assign(reason="unparseable_or_standardization_failed")); dec = dec[~bad]
inorg = ~dec.has_carbon
removed.append(dec[inorg].assign(reason="no_carbon_inorganic")); dec = dec[~inorg]
small = dec.n_heavy < 2
removed.append(dec[small].assign(reason="fewer_than_2_heavy_atoms")); dec = dec[~small]

# Duplicates: identical 2D structure (non-stereo standardized full SMILES) -> identical model
# input.  Merge if label range <= 1.0 log unit (mean), otherwise drop the whole set.
g = dec.groupby("std_nonstereo")
rng = g.Solubility.transform(lambda x: x.max() - x.min())
cnt = g.Solubility.transform("size")
conflict = (cnt > 1) & (rng > 1.0)
removed.append(dec[conflict].assign(reason="duplicate_2D_structure_label_conflict_gt_1logunit"))
dec = dec[~conflict]
log["dup_groups_total"] = int((dec.groupby("std_nonstereo").size() > 1).sum())
agg = dec.sort_values("ID").groupby("std_nonstereo", as_index=False).agg(
    ID=("ID", lambda x: "|".join(x)), Name=("Name", "first"), InChIKey=("InChIKey", "first"),
    SMILES=("SMILES", "first"), std_smiles=("std_smiles", "first"),
    parent_smiles=("parent_smiles", "first"), parent_ik=("parent_ik", "first"),
    parent_block1=("parent_block1", "first"), n_components=("n_components", "first"),
    has_ring_parent=("has_ring_parent", "first"), n_merged=("Solubility", "size"),
    label_range=("Solubility", lambda x: x.max() - x.min()), LogS=("Solubility", "mean"))
agg["is_multicomponent"] = agg.n_components > 1
agg = agg.sort_values("ID").reset_index(drop=True)
agg.insert(0, "mol_id", [f"M{i:05d}" for i in range(len(agg))])
rem = pd.concat(removed)
rem.to_csv(D / "04_removed_curation.csv", index=False)
agg.to_csv(D / "04_aqsoldb_curated.csv", index=False)
log.update({f"removed_{k}": int(v) for k, v in rem.reason.value_counts().items()})
log["merged_rows"] = int((agg.n_merged > 1).sum())
log["final_curated"] = len(agg)
log["final_multicomponent"] = int(agg.is_multicomponent.sum())
log["final_acyclic_parent"] = int((~agg.has_ring_parent).sum())
log["parent_block1_shared_by_multiple_entries"] = int((agg.parent_block1.value_counts() > 1).sum())
json.dump(log, open(R / "results/01_curation_log.json", "w"), indent=2)
print(json.dumps(log, indent=2))
