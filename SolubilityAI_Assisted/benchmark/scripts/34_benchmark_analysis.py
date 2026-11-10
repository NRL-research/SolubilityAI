"""
34_benchmark_analysis.py
========================
Interpretation layer: how benchmark performance compares with internal held-out test
performance, which compounds fail worst, and whether the failures share a structure.

This script analyses results. It does not change, re-tune or re-select any model.
Anything it suggests is written as a recommendation, not applied.

THE PATTERN TO EXPLAIN
----------------------
Every model shows a large POSITIVE mean signed error on every benchmark set -- it
predicts compounds as more soluble than the challenge measured them. The leading
hypothesis is a definitional mismatch rather than a modelling failure:

  * SC1 and SC2 report INTRINSIC solubility (logS0): the solubility of the neutral,
    un-ionised form of the compound.
  * AqSolDB aggregates general aqueous solubility measurements, many made at pH values
    where an ionisable compound is partly charged and therefore much more soluble.

For an ionisable drug the two quantities differ by orders of magnitude, always in the
same direction (logS >= logS0). A model trained on the first and scored against the
second would over-predict exactly as observed.

This script tests the hypothesis the obvious way: split each benchmark set into
compounds that carry an ionisable acidic or basic group and those that do not, and
compare the bias in each group. If the mismatch explanation holds, the ionisable
subset should carry substantially more positive bias. Labelled a HYPOTHESIS either
way -- a consistent result supports it but does not establish it.

Output: results/benchmark/34_generalisation.csv
        results/benchmark/34_worst_compounds.csv
        results/benchmark/34_analysis.json
"""

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

RES = "/home/claude/work/results/benchmark"
PRED = f"{RES}/predictions"
BDATA = "/home/claude/work/data/benchmark"

stats = pd.read_csv(f"{RES}/33_stats.csv")
info = json.load(open(f"{RES}/32_models_evaluated.json"))
HEADLINE = info["headline_models"]
SETS = list(info["sets"])
out = {}

# ---------------------------------------------------------------- 1. generalisation
gen = stats[["model", "set", "kind", "headline", "n", "rmse", "rmse_ci_low",
             "rmse_ci_high", "r2", "bias", "internal_test_rmse", "internal_test_r2",
             "delta_rmse_vs_internal_test"]].copy()


def verdict(row):
    if pd.isna(row["internal_test_rmse"]):
        return "no internal reference"
    lo, hi = row["rmse_ci_low"], row["rmse_ci_high"]
    t = row["internal_test_rmse"]
    if lo <= t <= hi:
        return "comparable (internal RMSE inside benchmark 95% CI)"
    return "degrades" if t < lo else "improves"


gen["generalisation"] = gen.apply(verdict, axis=1)
gen.to_csv(f"{RES}/34_generalisation.csv", index=False)

print("Generalisation vs internal held-out test (headline models)")
for s in SETS:
    sub = gen[(gen["set"] == s) & (gen["headline"])].sort_values("rmse")
    print(f"\n  {s} (n={sub['n'].iloc[0]})")
    for _, r in sub.iterrows():
        print(f"    {r['model']:22s} benchmark RMSE {r['rmse']:.3f} "
              f"[{r['rmse_ci_low']:.3f}, {r['rmse_ci_high']:.3f}]   "
              f"internal {r['internal_test_rmse']:.3f}   -> {r['generalisation']}")

out["generalisation_summary"] = json.loads(
    gen[gen["headline"]].to_json(orient="records"))

# ---------------------------------------------------------------- 2. rank stability
ranks = {}
for s in SETS:
    sub = stats[stats["set"] == s].sort_values("rmse")
    ranks[s] = {m: i + 1 for i, m in enumerate(sub["model"])}
internal_rank = {m: i + 1 for i, m in enumerate(
    stats.drop_duplicates("model").sort_values("internal_test_rmse")["model"])}
rk = pd.DataFrame({"internal": pd.Series(internal_rank), **{s: pd.Series(ranks[s]) for s in SETS}})
rk = rk.sort_values("internal")
out["rank_table"] = json.loads(rk.to_json(orient="index"))
from scipy import stats as sps
for s in SETS:
    rho = sps.spearmanr(rk["internal"], rk[s]).statistic
    out[f"rank_corr_internal_vs_{s}"] = float(rho)
print("\nRank agreement between internal test and each benchmark (Spearman over 14 models):")
for s in SETS:
    print(f"  {s:16s} rho = {out[f'rank_corr_internal_vs_{s}']:+.3f}")
print("\nRank table (1 = best):")
print(rk.to_string())

# ---------------------------------------------------------------- 3. ionisability test
def annotate(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    sulfonic = Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1,OX1-]")
    phenol = Chem.MolFromSmarts("[OX2H1][c]")
    basic_n = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N=*);!$([N+])]")
    arom_n = Chem.MolFromSmarts("[n;!$(n[#6]=O)]")
    has_acid = bool(m.GetSubstructMatches(acid) or m.GetSubstructMatches(sulfonic))
    has_base = bool(m.GetSubstructMatches(basic_n) or m.GetSubstructMatches(arom_n))
    return {
        "mw": Descriptors.MolWt(m),
        "clogp": Descriptors.MolLogP(m),
        "tpsa": Descriptors.TPSA(m),
        "n_rings": Descriptors.RingCount(m),
        "has_acidic_group": has_acid,
        "has_basic_group": has_base,
        "has_phenol": bool(m.GetSubstructMatches(phenol)),
        "ionisable": bool(has_acid or has_base),
    }


iontest = {}
for s in SETS:
    frames = []
    for m in HEADLINE:
        f = f"{PRED}/{m}__{s}.csv"
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
    if not frames:
        continue
    d = pd.concat(frames)
    ann = pd.DataFrame([annotate(x) for x in d["canonical_smiles"]], index=d.index)
    d = pd.concat([d, ann], axis=1)
    g = d.groupby("ionisable")["signed_error"].agg(["count", "mean", "std"])
    iontest[s] = json.loads(g.to_json(orient="index"))
    print(f"\n--- {s}: bias by ionisability (headline models pooled) ---")
    print(g.round(3).to_string())
    if True in g.index and False in g.index:
        diff = g.loc[True, "mean"] - g.loc[False, "mean"]
        print(f"    ionisable minus non-ionisable bias: {diff:+.3f} LogS")
        iontest[s]["bias_difference_ionisable_minus_not"] = float(diff)

out["ionisability_bias"] = iontest

# ---------------------------------------------------------------- 4. worst compounds
worst_rows = []
for s in SETS:
    for m in HEADLINE:
        f = f"{PRED}/{m}__{s}.csv"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        d["abs_error"] = d["signed_error"].abs()
        top = d.nlargest(5, "abs_error")
        ann = pd.DataFrame([annotate(x) for x in top["canonical_smiles"]], index=top.index)
        top = pd.concat([top, ann], axis=1)
        worst_rows.append(top)
worst = pd.concat(worst_rows)
cols = ["benchmark_set", "model", "name", "inchikey", "logS_exp", "logS_pred",
        "signed_error", "mw", "clogp", "ionisable", "has_acidic_group",
        "has_basic_group", "n_rings"]
worst[cols].to_csv(f"{RES}/34_worst_compounds.csv", index=False)

print("\n\nWorst-predicted compounds per set (headline models, union)")
for s in SETS:
    sub = worst[worst["benchmark_set"] == s]
    names = sub.groupby("name").agg(
        n_models=("model", "nunique"),
        mean_err=("signed_error", "mean"),
        logS_exp=("logS_exp", "first"),
        ionisable=("ionisable", "first"),
        mw=("mw", "first")).sort_values("mean_err", key=abs, ascending=False)
    print(f"\n  {s}:")
    print(names.round(2).head(8).to_string())

out["worst_compound_counts"] = {
    s: int(worst[worst["benchmark_set"] == s]["name"].nunique()) for s in SETS}

# ---------------------------------------------------------------- 5. range context
rng = {}
for s in SETS:
    d = pd.read_csv(f"{BDATA}/{s}.csv")
    y = pd.to_numeric(d["logS_exp"])
    rng[s] = {"n": int(len(y)), "min": float(y.min()), "max": float(y.max()),
              "sd": float(y.std()), "mean": float(y.mean())}
internal_y_sd = 2.5048      # SD of the internal held-out test set's LogS values
rng["internal_test"] = {"n": 1974, "sd": internal_y_sd}
out["target_ranges"] = rng
print("\n\nSpread of the experimental values (this is what R2 is measured against):")
for s, v in rng.items():
    if "min" in v:
        print(f"  {s:16s} n={v['n']:>3}  range [{v['min']:.2f}, {v['max']:.2f}]  SD {v['sd']:.2f}")
    else:
        print(f"  {s:16s} n={v['n']:>3}  SD {v['sd']:.2f}")

json.dump(out, open(f"{RES}/34_analysis.json", "w"), indent=2, default=str)
print(f"\nSaved {RES}/34_generalisation.csv, 34_worst_compounds.csv, 34_analysis.json")
