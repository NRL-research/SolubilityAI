"""
31_sc1_integrity.py
===================
SC1_processed.csv has a structural defect that would silently corrupt any evaluation
run on it. This script documents the defect, the evidence, and the two usable subsets
derived from it. It changes no source file.

THE DEFECT
----------
In SC1_processed.csv the experimental solubility travels with the `name` column, while
`cas_number`, `smiles`, `canonical_smiles` and `inchikey` travel together but belong to
a DIFFERENT compound for 8 of the 32 rows. Predicting the structure in a row and
scoring it against the solubility in the same row therefore compares a prediction for
one molecule with a measurement of another.

EVIDENCE (each pair independently checkable against published CAS numbers)
-------------------------------------------------------------------------
Rows 0-5 are offset: the name column has "2- Chloromandelic acid" sorted to the end of
that alphabetical block (a digit-after-letters sort) while the CAS/structure columns
keep the original sheet order, so each name sits one row ahead of its structure.

  row | name in file         | CAS in file  | CAS actually belongs to | structure in file
    0 | Acebutolol           | 10421-85-9   | 2-chloromandelic acid   | 2-chloromandelic acid
    1 | Amoxicillin          | 37517-30-9   | acebutolol              | acebutolol
    2 | Bendroflumethiazide  | 26787-78-0   | amoxicillin             | amoxicillin
    3 | Benzocaine           | 73-48-3      | bendroflumethiazide     | bendroflumethiazide
    4 | Benzthiazide         | 94-09-7      | benzocaine              | benzocaine
    5 | 2- Chloromandelic a. | 91-33-8      | benzthiazide            | benzthiazide

Rows 22-23 are transposed:
   22 | Pseudoephedrine      | 57-66-9      | probenecid              | probenecid
   23 | Probenecid           | 90-82-4      | pseudoephedrine         | (pseudo)ephedrine

WHICH COLUMN DOES THE MEASUREMENT FOLLOW?  The name.
Three independent checks, all agreeing:
  * Row 22 is named Pseudoephedrine and its value is "TSTM" (too soluble to measure).
    Pseudoephedrine is indeed freely soluble; probenecid -- the structure in that row --
    has an intrinsic solubility near LogS -4.6 and is certainly measurable.
  * Row 23 is named Probenecid with value -4.627, which matches probenecid's known
    intrinsic solubility, not pseudoephedrine's.
  * Row 4 is named Benzthiazide with value -4.829 (benzthiazide is poorly soluble)
    while its structure is benzocaine, whose intrinsic solubility is near -2.3 --
    and row 3, named Benzocaine, carries exactly -2.326.

CONSEQUENCES
------------
1. The leakage filter applied earlier in this project is UNAFFECTED. It used only the
   set of InChIKeys, and every structure in the file is a genuine SC1 compound; only
   the pairing with names and measurements is wrong. The 108 molecules removed from
   AqSolDB remain correctly removed.
2. Any evaluation using rows 0-5 or 22-23 as supplied is meaningless.
3. Four rows carry the non-numeric value "TSTM" and have no usable target at all.

TWO SUBSETS ARE PRODUCED
------------------------
  SC1_consistent (primary)  rows whose name, CAS, structure and measurement are all
                            mutually consistent, with TSTM rows dropped. Nothing is
                            inferred or repaired. This is what the headline SC1
                            numbers are computed on.
  SC1_repaired (secondary)  the full set with the offset undone -- each structure
                            paired with the measurement belonging to that structure's
                            own compound -- again with TSTM rows dropped. Reported as a
                            clearly-labelled sensitivity analysis, never as the
                            headline, because the repair rests on the inference above
                            rather than on the original SC1.xls source file.

Output: data/benchmark/SC1_consistent.csv, SC1_repaired.csv, 31_sc1_integrity.json
"""

import json
import os

import pandas as pd

UP = "/mnt/user-data/uploads/SolubilityAI_Assisted"
OUT = "/home/claude/work/data/benchmark"
RES = "/home/claude/work/results/benchmark"
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

TSTM = "TSTM"                         # "too soluble to measure" in the source challenge
OFFSET_BLOCK = list(range(0, 6))      # rows 0-5: name sits one row ahead of structure
SWAPPED_PAIR = [22, 23]               # rows 22-23: transposed

sc1 = pd.read_csv(f"{UP}/SC1_processed.csv")
sc1["row"] = sc1.index
sc1["logS_numeric"] = pd.to_numeric(sc1["logS0_uM"], errors="coerce")

report = {"source_rows": len(sc1)}

# ---------------------------------------------------------------- TSTM rows
tstm = sc1[sc1["logS_numeric"].isna()]
report["tstm_rows"] = json.loads(
    tstm[["row", "name", "cas_number", "logS0_uM"]].to_json(orient="records"))
report["n_tstm"] = len(tstm)
print(f"Rows with no numeric target (value '{TSTM}'): {len(tstm)}")
for _, r in tstm.iterrows():
    print(f"  row {r['row']:>2}  {r['name']}")

# ---------------------------------------------------------------- misaligned rows
mis = sorted(OFFSET_BLOCK + SWAPPED_PAIR)
report["misaligned_rows"] = json.loads(
    sc1.loc[mis, ["row", "name", "cas_number", "logS0_uM", "canonical_smiles"]]
    .to_json(orient="records"))
report["n_misaligned"] = len(mis)
print(f"\nRows where the structure does not belong to the named/measured compound: {len(mis)}")
for i in mis:
    print(f"  row {i:>2}  name={sc1.loc[i,'name']!r}  cas={sc1.loc[i,'cas_number']}")

# ---------------------------------------------------------------- subset 1: consistent
keep = [i for i in sc1.index if i not in mis and pd.notna(sc1.loc[i, "logS_numeric"])]
consistent = sc1.loc[keep, ["name", "cas_number", "inchikey", "canonical_smiles",
                            "logS_numeric", "row"]].copy()
consistent = consistent.rename(columns={"logS_numeric": "logS_exp"})
consistent["sd"] = pd.NA
consistent.to_csv(f"{OUT}/SC1_consistent.csv", index=False)
report["SC1_consistent_n"] = len(consistent)
print(f"\nSC1_consistent: {len(consistent)} compounds "
      f"({len(sc1)} rows - {len(mis)} misaligned - "
      f"{len(tstm) - len(set(mis) & set(tstm['row']))} remaining TSTM)")

# ---------------------------------------------------------------- subset 2: repaired
# Undo the offset: within rows 0-5 the structure in row i belongs with the measurement
# recorded one row earlier in name order, i.e. structure[i] pairs with target[(i-1) % 6].
# Within rows 22-23 the two are simply swapped.
target_row = {i: i for i in sc1.index}
for pos, i in enumerate(OFFSET_BLOCK):
    target_row[i] = OFFSET_BLOCK[(pos - 1) % len(OFFSET_BLOCK)]
target_row[22], target_row[23] = 23, 22

rep_rows = []
for i in sc1.index:
    t = target_row[i]
    if pd.isna(sc1.loc[t, "logS_numeric"]):
        continue
    rep_rows.append({
        "name": sc1.loc[t, "name"],                 # the compound the measurement is for
        "cas_number": sc1.loc[i, "cas_number"],     # consistent with the structure
        "inchikey": sc1.loc[i, "inchikey"],
        "canonical_smiles": sc1.loc[i, "canonical_smiles"],
        "logS_exp": float(sc1.loc[t, "logS_numeric"]),
        "row": i,
        "measurement_from_row": t,
        "sd": pd.NA,
    })
repaired = pd.DataFrame(rep_rows)
repaired.to_csv(f"{OUT}/SC1_repaired.csv", index=False)
report["SC1_repaired_n"] = len(repaired)
report["SC1_repair_mapping"] = {str(k): v for k, v in target_row.items() if k != v}
print(f"SC1_repaired:   {len(repaired)} compounds "
      f"({len(sc1)} rows - {len(tstm)} TSTM, offset undone)")

# ---------------------------------------------------------------- published size note
report["published_size_note"] = (
    "The project's own extraction notebook reports 35 compounds on the SC1 'Test Set' "
    "sheet; SC1_processed.csv contains 32, the other 3 having failed PubChem "
    "resolution and been dropped at extraction time. Of the 32, 4 carry 'TSTM' and 8 "
    "have a structure that does not match the measurement.")
report["leakage_filter_unaffected"] = (
    "The earlier SC leakage removal used only the set of InChIKeys. Every structure in "
    "the file is a genuine SC1 compound -- only the pairing with names and "
    "measurements is wrong -- so all 108 molecules removed from AqSolDB remain "
    "correctly removed.")

with open(f"{RES}/31_sc1_integrity.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nSaved {OUT}/SC1_consistent.csv, {OUT}/SC1_repaired.csv, "
      f"{RES}/31_sc1_integrity.json")

# ---------------------------------------------------------------- SC2 passthrough
for nm, fn in [("SC2_tight", "SC2_tight_processed.csv"),
               ("SC2_loose", "SC2_loose_processed.csv")]:
    d = pd.read_csv(f"{UP}/{fn}")
    out = d[["name", "inchikey", "canonical_smiles", "logS0", "sd", "n_measurements"]].copy()
    out = out.rename(columns={"logS0": "logS_exp"})
    out["logS_exp"] = pd.to_numeric(out["logS_exp"], errors="coerce")
    assert out["logS_exp"].notna().all(), f"{nm} has non-numeric targets"
    out.to_csv(f"{OUT}/{nm}.csv", index=False)
    print(f"{nm}: {len(out)} compounds, all targets numeric, "
          f"median measurement SD {out['sd'].median():.3f}")
