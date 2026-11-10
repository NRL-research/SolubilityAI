"""
01_sc_leakage_filter.py
=======================
Produce an SC-leakage-filtered version of AqSolDB.

Method follows the established project protocol (SolubilityAI History, Sections 8-11;
implemented originally in solubility_ai_v2.ipynb cells 18-21):

  1. Canonicalize every AqSolDB SMILES with RDKit (Chem.MolToSmiles) and compute a
     full standard InChIKey (Chem.inchi.MolToInchiKey). Drop unparseable rows.
  2. Remove every AqSolDB row whose InChIKey appears in the union of the InChIKeys of
     SC1, SC2-tight and SC2-loose (all three sets -- stricter than DISSOLVR, which
     removed SC2-tight only).
  3. Deduplicate the remainder by canonical SMILES, averaging Solubility (LogS).

Inputs  : AqSolDB.csv, SC1_processed.csv, SC2_tight_processed.csv, SC2_loose_processed.csv
Outputs : data/v1_sc_filtered/aqsoldb_sc_filtered.csv          (the deliverable)
          data/v1_sc_filtered/aqsoldb_canonicalized_raw.csv    (step-1 intermediate)
          data/v1_sc_filtered/sc_removed_molecules.csv         (audit trail of removals)
          data/v1_sc_filtered/01_counts.json                   (counts at every step)

Deterministic: no stochastic step. RDKit version recorded in the counts JSON.
"""

import json
import os
import sys

import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

UPLOADS = "/mnt/user-data/uploads/SolubilityAI_Assisted"
OUTDIR = "/home/claude/work/data/v1_sc_filtered"
os.makedirs(OUTDIR, exist_ok=True)


def canonicalize_and_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    canon = Chem.MolToSmiles(mol)
    try:
        ikey = MolToInchiKey(mol)
    except Exception:
        ikey = None
    return canon, ikey


def process_dataset(df, smiles_col, label=""):
    results = df[smiles_col].apply(canonicalize_and_inchikey)
    df = df.copy()
    df["canonical_smiles"] = [r[0] for r in results]
    df["inchikey"] = [r[1] for r in results]
    n_before = len(df)
    df = df.dropna(subset=["canonical_smiles", "inchikey"]).reset_index(drop=True)
    print(f"  [{label}] {n_before} -> {len(df)} after removing invalid/unparseable SMILES")
    return df, n_before


counts = {"rdkit_version": rdkit.__version__, "python": sys.version.split()[0]}

# ---------------------------------------------------------------- step 1
print("Step 1: canonicalize AqSolDB + compute InChIKeys")
aqsol_raw = pd.read_csv(f"{UPLOADS}/AqSolDB.csv")
counts["aqsoldb_rows_raw"] = len(aqsol_raw)
aqsol, n_before = process_dataset(aqsol_raw, "SMILES", "AqSolDB")
counts["aqsoldb_rows_after_rdkit_parse"] = len(aqsol)
counts["aqsoldb_rows_dropped_unparseable"] = n_before - len(aqsol)
aqsol.to_csv(f"{OUTDIR}/aqsoldb_canonicalized_raw.csv", index=False)

# ---------------------------------------------------------------- SC sets
print("\nLoading SC benchmark sets (already resolved to canonical SMILES + InChIKey)")
sc1 = pd.read_csv(f"{UPLOADS}/SC1_processed.csv")
sc2t = pd.read_csv(f"{UPLOADS}/SC2_tight_processed.csv")
sc2l = pd.read_csv(f"{UPLOADS}/SC2_loose_processed.csv")
for name, d in [("SC1", sc1), ("SC2-tight", sc2t), ("SC2-loose", sc2l)]:
    print(f"  {name}: {len(d)} rows, {d['inchikey'].nunique()} unique InChIKeys")
counts["sc1_rows"] = len(sc1)
counts["sc2_tight_rows"] = len(sc2t)
counts["sc2_loose_rows"] = len(sc2l)

sc1_k, sc2t_k, sc2l_k = set(sc1["inchikey"]), set(sc2t["inchikey"]), set(sc2l["inchikey"])
all_sc = sc1_k | sc2t_k | sc2l_k
counts["sc_unique_inchikeys_total"] = len(all_sc)
counts["sc1_unique_inchikeys"] = len(sc1_k)
counts["sc2_tight_unique_inchikeys"] = len(sc2t_k)
counts["sc2_loose_unique_inchikeys"] = len(sc2l_k)
counts["sc_overlap_sc1_sc2tight"] = len(sc1_k & sc2t_k)
counts["sc_overlap_sc1_sc2loose"] = len(sc1_k & sc2l_k)
counts["sc_overlap_sc2tight_sc2loose"] = len(sc2t_k & sc2l_k)
print(f"  union of all three: {len(all_sc)} unique InChIKeys")

# ---------------------------------------------------------------- step 2
print("\nStep 2: remove SC-overlapping molecules from AqSolDB by InChIKey")
mask = aqsol["inchikey"].isin(all_sc)
removed = aqsol[mask].copy()
removed["matched_SC1"] = removed["inchikey"].isin(sc1_k)
removed["matched_SC2_tight"] = removed["inchikey"].isin(sc2t_k)
removed["matched_SC2_loose"] = removed["inchikey"].isin(sc2l_k)
removed.to_csv(f"{OUTDIR}/sc_removed_molecules.csv", index=False)

clean = aqsol[~mask].reset_index(drop=True)
counts["aqsoldb_rows_removed_as_sc_leakage"] = int(mask.sum())
counts["aqsoldb_unique_inchikeys_removed"] = int(removed["inchikey"].nunique())
counts["removed_matched_sc1"] = int(removed["matched_SC1"].sum())
counts["removed_matched_sc2_tight"] = int(removed["matched_SC2_tight"].sum())
counts["removed_matched_sc2_loose"] = int(removed["matched_SC2_loose"].sum())
counts["sc_inchikeys_with_no_aqsoldb_match"] = int(len(all_sc - set(aqsol["inchikey"])))
counts["aqsoldb_rows_after_sc_removal"] = len(clean)
print(f"  AqSolDB: {len(aqsol)} -> {len(clean)} ({int(mask.sum())} rows removed)")
print(f"  distinct InChIKeys removed: {removed['inchikey'].nunique()}")

# ---------------------------------------------------------------- step 3
print("\nStep 3: deduplicate by canonical SMILES (mean LogS)")
n_pre = len(clean)
clean_dedup = clean.groupby("canonical_smiles", as_index=False).agg(
    {"inchikey": "first", "Solubility": "mean", "ID": "first", "Name": "first"}
)
counts["aqsoldb_rows_after_dedup"] = len(clean_dedup)
counts["aqsoldb_rows_collapsed_by_dedup"] = n_pre - len(clean_dedup)
print(f"  {n_pre} -> {len(clean_dedup)} ({n_pre - len(clean_dedup)} duplicate rows collapsed)")

out = f"{OUTDIR}/aqsoldb_sc_filtered.csv"
clean_dedup.to_csv(out, index=False)

# ---------------------------------------------------------------- verification
print("\nVerification:")
assert clean_dedup["canonical_smiles"].is_unique, "canonical SMILES not unique after dedup"
leak = set(clean_dedup["inchikey"]) & all_sc
print(f"  residual SC InChIKeys in output: {len(leak)} (must be 0)")
assert len(leak) == 0
counts["verification_residual_sc_inchikeys"] = 0
counts["output_file"] = out
counts["output_rows"] = len(clean_dedup)

with open(f"{OUTDIR}/01_counts.json", "w") as f:
    json.dump(counts, f, indent=2)
print(json.dumps(counts, indent=2))
