"""External benchmark evaluation, step 4: write results/external/ext_section.md (appended to REPORT.md)
from ext_report_rows.json.  No computation beyond formatting."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1]; X = R / "results/external"
E = json.load(open(X / "ext_report_rows.json")); V = E["validity"]; A = E["analysis"]
g = lambda m, k: next(r for r in E["rows"] if r["model"] == m and r["set"] == k)
H = ["blend_nnls", "blend_nnls_family", "r2_blend_nnls_ergfix"]
N = {"SC1": "SC1 (name-corrected)", "SC2_tight": "SC2-tight", "SC2_loose": "SC2-loose", "SC1_as_provided": "SC1 as provided (misaligned)"}
def tab(k, models):
    L = [f"| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = g(m, k)
        L.append(f"| `{m}` | {r['n']} | {r['rmse']:.3f} [{r['rmse_ci'][0]:.2f}, {r['rmse_ci'][1]:.2f}] | {r['mae']:.3f} | {r['r2']:.3f} [{r['r2_ci'][0]:.2f}, {r['r2_ci'][1]:.2f}] | "
                 f"{r['bias']:+.2f} | {r['pearson']:.2f} | {r['spearman']:.2f} | {r['w05']:.0%} | {r['w10']:.0%} | {r['test_rmse']:.3f} |")
    return "\n".join(L)
allm = lambda k: [r["model"] for r in sorted([r for r in E["rows"] if r["set"] == k], key=lambda r: r["rmse"])]
ion = [d for d in A["ionization"] if d["model"] == "blend_nnls"]
sim = [d for d in A["similarity"] if d["model"] == "blend_nnls"]
md = f"""

---

# External benchmark evaluation (2026-10-11)

This is a fixed, one-time evaluation of already-saved models on SC1, SC2-tight and SC2-loose.
**No model was trained, tuned, re-weighted or swapped.** Every model was loaded fresh in its
own process from its saved folder.

Scripts: `scripts/ext_01_benchmark_eval.py` (validity checks, predictions, statistics),
`ext_02_analysis.py` (interpretation), `ext_03_report_rows.py`, `ext_04_report_markdown.py`.
Outputs are in `results/external/`: per-compound files `pred_<model>__<set>.csv`, plus
`ext_metrics.csv`, `ext_validity.json`, `ext_sanity.json`, `ext_analysis.json` and
`ext_worst_compounds.csv`. Bootstrap CIs use 2000 compound resamples, seed 20261011. Sets are
never pooled.

**Models evaluated.** Headline: `blend_nnls` (confirmed best), `blend_nnls_family` (UI
recommendation) and `r2_blend_nnls_ergfix` (best reproducible round-2 model). All 21 of their
components were also evaluated (24 models in total; all loaded successfully). `r2_blend_nnls`
was not evaluated, because it is superseded by its reproducible rebuild.

## Validity checks
1. **Training overlap (InChIKey, connectivity block, parent block):** {"none for any of the 24 models in any set" if not any(V["any_overlap"].values()) else "OVERLAP FOUND"}.
   There is also no overlap with our test set. The training sets checked were v1 and v2, plus
   the in-domain rows used to fit the blend weights.
2. **Use in decisions:** the SC files are referenced in {", ".join(f"`{f}`" for f in V["code_refs"])}.
   They were used only for decontamination (identities, not values), leakage audits, and a
   secondary external check reported after each round's pre-registered test evaluation. No
   selection, tuning, early stopping, blend weight or domain rule used SC values. Caveat:
   round-1 SC numbers were seen before round 2 started, but round 2 was driven by
   training-set error analysis.
3. **Units:** SC1's `logS0_uM` spans {V["units"]["SC1"]["min"]:.2f} to {V["units"]["SC1"]["max"]:.2f}.
   As log10(µM), salicylic acid (−1.93) would be about 2 ng/mL, roughly 10⁶ below its known
   solubility; as log10(mol/L) it matches. SC2 `logS0` is also log10(mol/L). **No conversion
   was applied.**
4. **Counts:**
   - SC1: 32 rows (= the published test set); {len(V["excluded"]["SC1"])} are "TSTM" (too soluble to measure: {", ".join(e["name"].strip() for e in V["excluded"]["SC1"])}) and cannot be scored, giving **n = {V["counts"]["SC1"]["evaluable"]}**.
   - SC2-tight: **n = {V["counts"]["SC2_tight"]["evaluable"]}**/100.
   - SC2-loose: **n = {V["counts"]["SC2_loose"]["evaluable"]}**/32.
   - No parse failures; all recomputed InChIKeys match the files.
   - **SC1's 8 misaligned rows were re-aligned to their names** (§1.1). After correction, the
     shared compounds agree with SC2-tight: benzthiazide −4.83 vs −4.84, probenecid −4.63 vs
     −4.83, folic acid −5.25 vs −5.96. The exception is indomethacin, −2.94 vs −5.48
     (unchanged, flagged).
5. **Sanity:**
   - No two models gave identical predictions.
   - Each blend equals the weighted sum of its components (≤ 2e-15).
   - The 7 compounds shared between sets got identical predictions in each set.

## Results (headline models)
### {N["SC1"]}
{tab("SC1", H)}

### {N["SC2_tight"]}
{tab("SC2_tight", H)}

### {N["SC2_loose"]}
{tab("SC2_loose", H)}

**Measurement noise (SC2 per-compound SD):**
- SC2-tight: RMS SD {g("blend_nnls","SC2_tight")["noise"]:.2f}, which is negligible next to the RMSE of {g("blend_nnls","SC2_tight")["rmse"]:.2f}.
- SC2-loose: RMS SD {g("blend_nnls","SC2_loose")["noise"]:.2f}, about {g("blend_nnls","SC2_loose")["noise"]**2/g("blend_nnls","SC2_loose")["rmse"]**2:.0%} of the mean squared error. The model-attributable RMSE is about {g("blend_nnls","SC2_loose")["beyond"]:.2f}.
- Only {g("blend_nnls","SC2_loose")["w2sd"]:.0%} (SC2-loose) and {g("blend_nnls","SC2_tight")["w2sd"]:.0%} (SC2-tight) of compounds are within 2 SD of their measurement.

## Interpretation
* **The models degrade relative to our own test set.** RMSE is about 1.2× higher on SC2-tight,
  1.4× on SC1 and 2.1× on SC2-loose. The three headline models stay within about 0.05 RMSE of
  each other, so the benchmarks cannot rank them.
* **There is a systematic positive bias** (predicted more soluble than measured), making up
  24–37% of the MSE. It is larger for acids and bases than for neutral compounds (blend_nnls,
  SC2-tight: acids {next(d for d in ion if d["set"]=="SC2_tight" and d["ion_class"]=="acid")["bias"]:+.2f},
  bases {next(d for d in ion if d["set"]=="SC2_tight" and d["ion_class"]=="base")["bias"]:+.2f},
  neutral/weak {next(d for d in ion if d["set"]=="SC2_tight" and d["ion_class"]=="neutral/weak")["bias"]:+.2f}).
  *Hypothesis:* the benchmarks report intrinsic solubility, whereas AqSolDB's are largely
  apparent solubilities of partly ionised compounds.
* **Novelty matters.** Median nearest-neighbour similarity is {A["range_restriction"]["SC1"]["median_nn_sim"]:.2f} for SC1, {A["range_restriction"]["SC2_tight"]["median_nn_sim"]:.2f} for SC2-tight and {A["range_restriction"]["SC2_loose"]["median_nn_sim"]:.2f} for SC2-loose.
  On SC2-tight, RMSE is {next(d for d in sim if d["set"]=="SC2_tight" and d["band"]=="0.6-1.0")["rmse"]:.2f} at similarity ≥ 0.6 against {next(d for d in sim if d["set"]=="SC2_tight" and d["band"]=="0-0.4")["rmse"]:.2f} below 0.4.
* **The worst errors are very insoluble, large, lipophilic drugs predicted too soluble** (17 of
  the 18 worst). Examples: amiodarone, clofazimine, itraconazole, terfenadine, tamoxifen,
  cisapride and aripiprazole.
* **Range restriction.** SC1 and SC2-tight have narrow value ranges (SD {A["range_restriction"]["SC1"]["benchmark_logS_sd"]:.2f} and {A["range_restriction"]["SC2_tight"]["benchmark_logS_sd"]:.2f}, against {A["own_test_logS_sd"]:.2f} in our test set). Even at its test RMSE, R² on SC2-tight would be only about {A["range_restriction"]["SC2_tight"]["r2_if_rmse_equal_to_test"]:.2f}.
* **Limits of what can be concluded.** n = 28, 100 and 32; the CIs are wide and R² is unstable.
* **Recommendations (not applied):**
  - show nearest-neighbour similarity in any UI and warn below 0.4;
  - state that predictions are AqSolDB-style water solubility, not intrinsic solubility;
  - re-check the SC1 rows against the original publication.

## Appendix: all evaluated models
"""
for k in ["SC1", "SC2_tight", "SC2_loose", "SC1_as_provided"]:
    md += f"\n### {N[k]}\n" + tab(k, allm(k)) + "\n"
open(X / "ext_section.md", "w").write(md)
print(len(md))
