# Data

## Raw Data

The raw AqSolDB dataset isn't downloaded fresh anymore — `common/rebuild_scaffold_split.py` reconstructs the full working dataset from `misc/original_buggy_split/data_processed/{train,test}.csv` (the original committed split, which per the provenance note below already is the canonicalized AqSolDB with invalid SMILES dropped). If you need the truly raw file for some other purpose:

**AqSolDB** (Sorkun et al., 2019 — *Scientific Data*)
- URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8
- Columns: `ID`, `Name`, `InChIKey`, `SMILES`, `Solubility`
- Size: ~9,982 compounds

---

## Processed Splits — Committed

| File | Description | Size |
|---|---|---|
| `processed_scaffold_fixed/train.csv` | Corrected scaffold split — training set | 6,066 molecules |
| `processed_scaffold_fixed/val.csv` | Corrected scaffold split — validation set | 1,918 molecules |
| `processed_scaffold_fixed/test.csv` | Corrected scaffold split — test set | 1,996 molecules |
| `processed_random_fixed/{train,val,test}.csv` | Corrected random split, same sizes | — |

All splits: `random_seed=42`, `MurckoScaffold` algorithm (RDKit), with acyclic (ring-free) molecules given individual split keys rather than one shared block — see the main README's Methodology Correction for why that matters.

Additional columns beyond the raw AqSolDB:
- `canonical_smiles` — RDKit-canonicalized SMILES (used instead of raw `SMILES` for all modeling)
- `scaffold` — Murcko scaffold SMILES (used for scaffold grouping; empty string for acyclic molecules)

---

## Reproducing the Splits

```bash
# From the repository root:
python common/rebuild_scaffold_split.py
```

---

## Data Provenance

- Original source: Sorkun MC, Khetan A, Er S (2019). AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. *Scientific Data* 6, 143. https://doi.org/10.1038/s41597-019-0151-1
- Solubility unit: LogS = log₁₀(S), where S is in mol/L. Units originally in g/L and mg/L were converted using molecular weight.
- No additional curation was applied beyond canonicalization and invalid SMILES removal (2 molecules dropped).
