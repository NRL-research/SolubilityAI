

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
1. **Training overlap (InChIKey, connectivity block, parent block):** none for any of the 24 models in any set.
   There is also no overlap with our test set. The training sets checked were v1 and v2, plus
   the in-domain rows used to fit the blend weights.
2. **Use in decisions:** the SC files are referenced in `scripts/01_curate.py`, `scripts/07_test_eval.py`, `scripts/08_verify.py`, `scripts/09_write_model_docs.py`, `scripts/r2_07_test_eval.py`, `scripts/r2_08_verify.py`, `scripts/r2_08b_verify_ergfix.py`.
   They were used only for decontamination (identities, not values), leakage audits, and a
   secondary external check reported after each round's pre-registered test evaluation. No
   selection, tuning, early stopping, blend weight or domain rule used SC values. Caveat:
   round-1 SC numbers were seen before round 2 started, but round 2 was driven by
   training-set error analysis.
3. **Units:** SC1's `logS0_uM` spans -7.74 to -1.87.
   As log10(µM), salicylic acid (−1.93) would be about 2 ng/mL, roughly 10⁶ below its known
   solubility; as log10(mol/L) it matches. SC2 `logS0` is also log10(mol/L). **No conversion
   was applied.**
4. **Counts:**
   - SC1: 32 rows (= the published test set); 4 are "TSTM" (too soluble to measure: 2- Chloromandelic acid, Ephedrine, Marbofloxacin, Pseudoephedrine) and cannot be scored, giving **n = 28**.
   - SC2-tight: **n = 100**/100.
   - SC2-loose: **n = 32**/32.
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
### SC1 (name-corrected)
| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `blend_nnls` | 28 | 1.154 [0.88, 1.43] | 0.912 | 0.257 [-0.40, 0.51] | +0.70 | 0.73 | 0.70 | 32% | 61% | 0.799 |
| `blend_nnls_family` | 28 | 1.156 [0.87, 1.43] | 0.915 | 0.254 [-0.39, 0.51] | +0.70 | 0.73 | 0.70 | 29% | 61% | 0.796 |
| `r2_blend_nnls_ergfix` | 28 | 1.172 [0.88, 1.45] | 0.925 | 0.233 [-0.45, 0.51] | +0.71 | 0.72 | 0.66 | 36% | 61% | 0.793 |

### SC2-tight
| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `blend_nnls` | 100 | 0.983 [0.83, 1.14] | 0.766 | 0.397 [0.14, 0.56] | +0.48 | 0.74 | 0.70 | 41% | 74% | 0.799 |
| `blend_nnls_family` | 100 | 0.994 [0.84, 1.15] | 0.771 | 0.384 [0.12, 0.56] | +0.48 | 0.73 | 0.70 | 41% | 74% | 0.796 |
| `r2_blend_nnls_ergfix` | 100 | 0.979 [0.83, 1.13] | 0.759 | 0.402 [0.15, 0.56] | +0.50 | 0.75 | 0.72 | 39% | 71% | 0.793 |

### SC2-loose
| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `blend_nnls` | 32 | 1.685 [1.16, 2.26] | 1.277 | 0.381 [-0.02, 0.63] | +1.02 | 0.79 | 0.75 | 25% | 47% | 0.799 |
| `blend_nnls_family` | 32 | 1.729 [1.19, 2.33] | 1.310 | 0.348 [-0.08, 0.61] | +1.05 | 0.77 | 0.75 | 22% | 47% | 0.796 |
| `r2_blend_nnls_ergfix` | 32 | 1.698 [1.17, 2.30] | 1.282 | 0.371 [-0.05, 0.63] | +1.01 | 0.78 | 0.75 | 28% | 50% | 0.793 |

**Measurement noise (SC2 per-compound SD):**
- SC2-tight: RMS SD 0.17, which is negligible next to the RMSE of 0.98.
- SC2-loose: RMS SD 0.63, about 14% of the mean squared error. The model-attributable RMSE is about 1.56.
- Only 53% (SC2-loose) and 27% (SC2-tight) of compounds are within 2 SD of their measurement.

## Interpretation
* **The models degrade relative to our own test set.** RMSE is about 1.2× higher on SC2-tight,
  1.4× on SC1 and 2.1× on SC2-loose. The three headline models stay within about 0.05 RMSE of
  each other, so the benchmarks cannot rank them.
* **There is a systematic positive bias** (predicted more soluble than measured), making up
  24–37% of the MSE. It is larger for acids and bases than for neutral compounds (blend_nnls,
  SC2-tight: acids +0.76,
  bases +0.93,
  neutral/weak +0.25).
  *Hypothesis:* the benchmarks report intrinsic solubility, whereas AqSolDB's are largely
  apparent solubilities of partly ionised compounds.
* **Novelty matters.** Median nearest-neighbour similarity is 0.54 for SC1, 0.56 for SC2-tight and 0.37 for SC2-loose.
  On SC2-tight, RMSE is 0.60 at similarity ≥ 0.6 against 1.32 below 0.4.
* **The worst errors are very insoluble, large, lipophilic drugs predicted too soluble** (17 of
  the 18 worst). Examples: amiodarone, clofazimine, itraconazole, terfenadine, tamoxifen,
  cisapride and aripiprazole.
* **Range restriction.** SC1 and SC2-tight have narrow value ranges (SD 1.34 and 1.27, against 2.31 in our test set). Even at its test RMSE, R² on SC2-tight would be only about 0.60.
* **Limits of what can be concluded.** n = 28, 100 and 32; the CIs are wide and R² is unstable.
* **Recommendations (not applied):**
  - show nearest-neighbour similarity in any UI and warn below 0.4;
  - state that predictions are AqSolDB-style water solubility, not intrinsic solubility;
  - re-check the SC1 rows against the original publication.

## Appendix: all evaluated models

### SC1 (name-corrected)
| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `xgb_desc_morgan_v2` | 28 | 1.072 [0.82, 1.32] | 0.853 | 0.359 [-0.36, 0.65] | +0.60 | 0.75 | 0.71 | 32% | 61% | 0.844 |
| `lgbm_mordred` | 28 | 1.074 [0.84, 1.29] | 0.873 | 0.357 [-0.40, 0.65] | +0.66 | 0.78 | 0.66 | 39% | 64% | 0.845 |
| `r2_lgbm_desc_mordred_avalon_erg2` | 28 | 1.111 [0.82, 1.40] | 0.887 | 0.311 [-0.59, 0.66] | +0.72 | 0.77 | 0.67 | 32% | 64% | 0.819 |
| `r2_lgbm_d_av_erg2` | 28 | 1.127 [0.82, 1.40] | 0.877 | 0.291 [-0.50, 0.59] | +0.67 | 0.73 | 0.67 | 36% | 71% | 0.822 |
| `lgbm_desc_mordred` | 28 | 1.131 [0.81, 1.45] | 0.877 | 0.286 [-0.61, 0.65] | +0.72 | 0.76 | 0.64 | 32% | 64% | 0.832 |
| `gnn_hybrid_s43` | 28 | 1.144 [0.76, 1.49] | 0.838 | 0.269 [-0.54, 0.61] | +0.55 | 0.67 | 0.69 | 46% | 75% | 0.922 |
| `r2_svr_dm` | 28 | 1.151 [0.90, 1.41] | 0.944 | 0.261 [-0.57, 0.59] | +0.66 | 0.71 | 0.65 | 32% | 57% | 0.957 |
| `blend_nnls` | 28 | 1.154 [0.88, 1.43] | 0.912 | 0.257 [-0.40, 0.51] | +0.70 | 0.73 | 0.70 | 32% | 61% | 0.799 |
| `blend_nnls_family` | 28 | 1.156 [0.87, 1.43] | 0.915 | 0.254 [-0.39, 0.51] | +0.70 | 0.73 | 0.70 | 29% | 61% | 0.796 |
| `lgbm_desc_v2` | 28 | 1.159 [0.84, 1.49] | 0.916 | 0.251 [-0.72, 0.62] | +0.67 | 0.71 | 0.60 | 32% | 61% | 0.847 |
| `gnn_hybrid_s44` | 28 | 1.161 [0.78, 1.53] | 0.907 | 0.248 [-0.36, 0.54] | +0.64 | 0.69 | 0.73 | 29% | 71% | 0.910 |
| `r2_lgbm_dm_huber` | 28 | 1.166 [0.85, 1.48] | 0.915 | 0.241 [-0.71, 0.63] | +0.74 | 0.74 | 0.64 | 36% | 64% | 0.834 |
| `r2_lgbm_dm_m2v` | 28 | 1.167 [0.83, 1.49] | 0.914 | 0.240 [-0.80, 0.65] | +0.73 | 0.74 | 0.55 | 39% | 61% | 0.834 |
| `r2_blend_nnls_ergfix` | 28 | 1.172 [0.88, 1.45] | 0.925 | 0.233 [-0.45, 0.51] | +0.71 | 0.72 | 0.66 | 36% | 61% | 0.793 |
| `lgbm_desc_maccs` | 28 | 1.192 [0.90, 1.47] | 0.944 | 0.207 [-0.62, 0.53] | +0.68 | 0.68 | 0.59 | 36% | 57% | 0.852 |
| `r2_catboost_dm` | 28 | 1.201 [0.89, 1.50] | 0.959 | 0.195 [-0.74, 0.55] | +0.78 | 0.73 | 0.63 | 32% | 61% | 0.833 |
| `svr_desc` | 28 | 1.270 [0.91, 1.61] | 0.975 | 0.101 [-0.64, 0.43] | +0.66 | 0.59 | 0.56 | 32% | 61% | 0.899 |
| `svr_desc_v2` | 28 | 1.302 [0.94, 1.65] | 1.011 | 0.054 [-0.74, 0.39] | +0.67 | 0.56 | 0.54 | 32% | 64% | 0.893 |
| `gnn_hybrid` | 28 | 1.347 [0.85, 1.83] | 0.935 | -0.013 [-0.79, 0.45] | +0.65 | 0.51 | 0.64 | 36% | 68% | 0.929 |
| `ridge_desc` | 28 | 1.352 [0.94, 1.73] | 0.997 | -0.020 [-1.08, 0.42] | +0.81 | 0.59 | 0.47 | 43% | 57% | 1.088 |
| `mlp_desc` | 28 | 1.399 [1.08, 1.71] | 1.163 | -0.093 [-1.12, 0.30] | +0.68 | 0.54 | 0.58 | 21% | 50% | 0.939 |
| `gnn_mpnn` | 28 | 1.418 [1.02, 1.83] | 1.111 | -0.123 [-0.88, 0.26] | +0.87 | 0.57 | 0.65 | 29% | 50% | 0.927 |
| `gnn_mpnn_s43` | 28 | 1.656 [1.13, 2.20] | 1.240 | -0.531 [-1.49, 0.01] | +0.96 | 0.31 | 0.34 | 25% | 57% | 0.955 |
| `knn_morganbits_k5` | 28 | 1.714 [1.18, 2.23] | 1.250 | -0.640 [-2.16, 0.06] | +0.72 | 0.18 | 0.16 | 43% | 57% | 1.605 |

### SC2-tight
| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `r2_catboost_dm` | 100 | 0.942 [0.79, 1.08] | 0.728 | 0.446 [0.19, 0.60] | +0.46 | 0.77 | 0.76 | 45% | 73% | 0.833 |
| `lgbm_mordred` | 100 | 0.945 [0.81, 1.08] | 0.731 | 0.443 [0.20, 0.59] | +0.47 | 0.76 | 0.75 | 41% | 73% | 0.845 |
| `r2_svr_dm` | 100 | 0.950 [0.80, 1.11] | 0.703 | 0.436 [0.18, 0.60] | +0.47 | 0.77 | 0.72 | 52% | 71% | 0.957 |
| `xgb_desc_morgan_v2` | 100 | 0.968 [0.84, 1.09] | 0.773 | 0.415 [0.19, 0.56] | +0.42 | 0.73 | 0.69 | 39% | 69% | 0.844 |
| `r2_lgbm_desc_mordred_avalon_erg2` | 100 | 0.979 [0.82, 1.13] | 0.743 | 0.402 [0.14, 0.58] | +0.50 | 0.75 | 0.74 | 45% | 73% | 0.819 |
| `r2_blend_nnls_ergfix` | 100 | 0.979 [0.83, 1.13] | 0.759 | 0.402 [0.15, 0.56] | +0.50 | 0.75 | 0.72 | 39% | 71% | 0.793 |
| `blend_nnls` | 100 | 0.983 [0.83, 1.14] | 0.766 | 0.397 [0.14, 0.56] | +0.48 | 0.74 | 0.70 | 41% | 74% | 0.799 |
| `r2_lgbm_dm_huber` | 100 | 0.984 [0.83, 1.13] | 0.761 | 0.395 [0.14, 0.57] | +0.46 | 0.73 | 0.73 | 43% | 72% | 0.834 |
| `svr_desc` | 100 | 0.993 [0.85, 1.13] | 0.781 | 0.385 [0.13, 0.56] | +0.44 | 0.72 | 0.69 | 40% | 71% | 0.899 |
| `blend_nnls_family` | 100 | 0.994 [0.84, 1.15] | 0.771 | 0.384 [0.12, 0.56] | +0.48 | 0.73 | 0.70 | 41% | 74% | 0.796 |
| `r2_lgbm_d_av_erg2` | 100 | 1.002 [0.84, 1.15] | 0.768 | 0.374 [0.10, 0.55] | +0.51 | 0.74 | 0.71 | 41% | 75% | 0.822 |
| `r2_lgbm_dm_m2v` | 100 | 1.003 [0.84, 1.16] | 0.763 | 0.372 [0.09, 0.54] | +0.49 | 0.73 | 0.71 | 45% | 71% | 0.834 |
| `lgbm_desc_mordred` | 100 | 1.019 [0.86, 1.18] | 0.784 | 0.353 [0.08, 0.53] | +0.48 | 0.71 | 0.70 | 40% | 72% | 0.832 |
| `lgbm_desc_maccs` | 100 | 1.030 [0.87, 1.19] | 0.794 | 0.338 [0.06, 0.51] | +0.52 | 0.71 | 0.71 | 43% | 73% | 0.852 |
| `lgbm_desc_v2` | 100 | 1.033 [0.88, 1.19] | 0.824 | 0.334 [0.06, 0.51] | +0.45 | 0.69 | 0.68 | 39% | 66% | 0.847 |
| `svr_desc_v2` | 100 | 1.036 [0.89, 1.18] | 0.812 | 0.331 [0.05, 0.51] | +0.47 | 0.70 | 0.66 | 38% | 71% | 0.893 |
| `gnn_hybrid_s43` | 100 | 1.123 [0.95, 1.31] | 0.868 | 0.213 [-0.15, 0.45] | +0.51 | 0.66 | 0.65 | 35% | 65% | 0.922 |
| `gnn_hybrid_s44` | 100 | 1.147 [0.96, 1.34] | 0.865 | 0.179 [-0.21, 0.43] | +0.59 | 0.66 | 0.62 | 39% | 69% | 0.910 |
| `gnn_mpnn` | 100 | 1.165 [1.00, 1.34] | 0.894 | 0.153 [-0.21, 0.39] | +0.54 | 0.64 | 0.61 | 35% | 68% | 0.927 |
| `mlp_desc` | 100 | 1.178 [1.01, 1.36] | 0.920 | 0.134 [-0.27, 0.40] | +0.53 | 0.64 | 0.62 | 37% | 64% | 0.939 |
| `gnn_hybrid` | 100 | 1.183 [0.99, 1.39] | 0.917 | 0.127 [-0.27, 0.39] | +0.55 | 0.62 | 0.58 | 32% | 65% | 0.929 |
| `ridge_desc` | 100 | 1.202 [1.03, 1.37] | 0.972 | 0.098 [-0.25, 0.33] | +0.64 | 0.61 | 0.61 | 28% | 66% | 1.088 |
| `gnn_mpnn_s43` | 100 | 1.213 [1.02, 1.42] | 0.901 | 0.081 [-0.34, 0.36] | +0.60 | 0.62 | 0.61 | 38% | 65% | 0.955 |
| `knn_morganbits_k5` | 100 | 1.474 [1.27, 1.68] | 1.132 | -0.356 [-0.93, 0.03] | +0.69 | 0.37 | 0.33 | 28% | 58% | 1.605 |

### SC2-loose
| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `xgb_desc_morgan_v2` | 32 | 1.480 [1.05, 1.91] | 1.146 | 0.522 [0.22, 0.70] | +0.88 | 0.85 | 0.82 | 31% | 56% | 0.844 |
| `r2_lgbm_dm_m2v` | 32 | 1.505 [1.02, 2.03] | 1.129 | 0.506 [0.18, 0.71] | +0.88 | 0.84 | 0.82 | 25% | 62% | 0.834 |
| `r2_lgbm_dm_huber` | 32 | 1.530 [1.10, 1.98] | 1.199 | 0.490 [0.17, 0.68] | +0.93 | 0.84 | 0.81 | 28% | 53% | 0.834 |
| `lgbm_desc_v2` | 32 | 1.536 [1.12, 1.98] | 1.224 | 0.485 [0.18, 0.67] | +0.86 | 0.81 | 0.81 | 22% | 47% | 0.847 |
| `lgbm_desc_maccs` | 32 | 1.556 [1.13, 2.00] | 1.213 | 0.472 [0.16, 0.66] | +0.96 | 0.84 | 0.83 | 22% | 53% | 0.852 |
| `lgbm_mordred` | 32 | 1.570 [1.08, 2.05] | 1.175 | 0.463 [0.14, 0.67] | +0.95 | 0.84 | 0.82 | 28% | 53% | 0.845 |
| `r2_lgbm_desc_mordred_avalon_erg2` | 32 | 1.592 [1.11, 2.10] | 1.207 | 0.448 [0.09, 0.67] | +0.98 | 0.83 | 0.81 | 25% | 53% | 0.819 |
| `r2_lgbm_d_av_erg2` | 32 | 1.653 [1.21, 2.13] | 1.333 | 0.404 [0.06, 0.61] | +1.01 | 0.80 | 0.80 | 16% | 44% | 0.822 |
| `r2_catboost_dm` | 32 | 1.661 [1.20, 2.15] | 1.314 | 0.398 [0.02, 0.62] | +0.95 | 0.78 | 0.72 | 19% | 50% | 0.833 |
| `blend_nnls` | 32 | 1.685 [1.16, 2.26] | 1.277 | 0.381 [-0.02, 0.63] | +1.02 | 0.79 | 0.75 | 25% | 47% | 0.799 |
| `r2_blend_nnls_ergfix` | 32 | 1.698 [1.17, 2.30] | 1.282 | 0.371 [-0.05, 0.63] | +1.01 | 0.78 | 0.75 | 28% | 50% | 0.793 |
| `blend_nnls_family` | 32 | 1.729 [1.19, 2.33] | 1.310 | 0.348 [-0.08, 0.61] | +1.05 | 0.77 | 0.75 | 22% | 47% | 0.796 |
| `lgbm_desc_mordred` | 32 | 1.735 [1.21, 2.32] | 1.318 | 0.344 [-0.10, 0.61] | +1.04 | 0.77 | 0.73 | 16% | 44% | 0.832 |
| `ridge_desc` | 32 | 1.765 [1.32, 2.18] | 1.353 | 0.321 [-0.15, 0.55] | +1.25 | 0.86 | 0.79 | 34% | 47% | 1.088 |
| `svr_desc_v2` | 32 | 1.869 [1.22, 2.55] | 1.369 | 0.238 [-0.25, 0.56] | +1.10 | 0.72 | 0.74 | 22% | 47% | 0.893 |
| `svr_desc` | 32 | 1.887 [1.21, 2.58] | 1.363 | 0.224 [-0.27, 0.57] | +1.09 | 0.70 | 0.74 | 28% | 50% | 0.899 |
| `r2_svr_dm` | 32 | 1.924 [1.23, 2.64] | 1.388 | 0.193 [-0.34, 0.56] | +1.18 | 0.72 | 0.74 | 25% | 44% | 0.957 |
| `gnn_mpnn_s43` | 32 | 1.970 [1.35, 2.56] | 1.391 | 0.154 [-0.52, 0.55] | +1.02 | 0.64 | 0.66 | 34% | 53% | 0.955 |
| `gnn_mpnn` | 32 | 2.042 [1.37, 2.76] | 1.516 | 0.091 [-0.57, 0.51] | +1.09 | 0.62 | 0.58 | 22% | 38% | 0.927 |
| `gnn_hybrid` | 32 | 2.060 [1.43, 2.69] | 1.536 | 0.075 [-0.61, 0.49] | +1.23 | 0.65 | 0.67 | 25% | 47% | 0.929 |
| `mlp_desc` | 32 | 2.121 [1.51, 2.74] | 1.594 | 0.019 [-0.58, 0.38] | +1.18 | 0.59 | 0.64 | 25% | 53% | 0.939 |
| `gnn_hybrid_s44` | 32 | 2.228 [1.42, 3.13] | 1.607 | -0.082 [-0.93, 0.46] | +1.17 | 0.52 | 0.54 | 22% | 50% | 0.910 |
| `gnn_hybrid_s43` | 32 | 2.231 [1.41, 3.12] | 1.553 | -0.085 [-0.98, 0.51] | +1.12 | 0.53 | 0.53 | 25% | 53% | 0.922 |
| `knn_morganbits_k5` | 32 | 2.782 [2.19, 3.39] | 2.257 | -0.688 [-1.84, -0.14] | +2.11 | 0.53 | 0.51 | 12% | 28% | 1.605 |

### SC1 as provided (misaligned)
| model | n | RMSE [95% CI] | MAE | R² [95% CI] | bias | Pearson | Spearman | ≤0.5 | ≤1.0 | own test RMSE |
|---|---|---|---|---|---|---|---|---|---|---|
| `lgbm_mordred` | 28 | 1.389 [0.96, 1.84] | 1.057 | -0.077 [-1.83, 0.55] | +0.80 | 0.59 | 0.49 | 39% | 61% | 0.845 |
| `xgb_desc_morgan_v2` | 28 | 1.409 [1.01, 1.79] | 1.118 | -0.108 [-1.73, 0.49] | +0.76 | 0.54 | 0.43 | 21% | 57% | 0.844 |
| `lgbm_desc_v2` | 28 | 1.456 [0.99, 1.91] | 1.109 | -0.182 [-2.03, 0.49] | +0.82 | 0.54 | 0.39 | 29% | 57% | 0.847 |
| `r2_lgbm_dm_huber` | 28 | 1.469 [1.03, 1.90] | 1.127 | -0.204 [-2.03, 0.48] | +0.88 | 0.57 | 0.43 | 32% | 61% | 0.834 |
| `r2_lgbm_desc_mordred_avalon_erg2` | 28 | 1.471 [1.03, 1.90] | 1.143 | -0.208 [-2.06, 0.48] | +0.87 | 0.56 | 0.39 | 29% | 54% | 0.819 |
| `r2_lgbm_d_av_erg2` | 28 | 1.472 [1.05, 1.86] | 1.125 | -0.209 [-1.93, 0.41] | +0.83 | 0.53 | 0.39 | 32% | 57% | 0.822 |
| `lgbm_desc_mordred` | 28 | 1.472 [1.01, 1.91] | 1.114 | -0.209 [-2.06, 0.47] | +0.88 | 0.56 | 0.40 | 29% | 54% | 0.832 |
| `blend_nnls` | 28 | 1.474 [1.04, 1.88] | 1.124 | -0.213 [-1.80, 0.38] | +0.85 | 0.52 | 0.49 | 29% | 57% | 0.799 |
| `r2_lgbm_dm_m2v` | 28 | 1.482 [1.01, 1.95] | 1.116 | -0.226 [-2.22, 0.50] | +0.87 | 0.56 | 0.35 | 36% | 57% | 0.834 |
| `blend_nnls_family` | 28 | 1.483 [1.06, 1.88] | 1.141 | -0.227 [-1.83, 0.36] | +0.86 | 0.51 | 0.48 | 25% | 57% | 0.796 |
| `lgbm_desc_maccs` | 28 | 1.497 [1.08, 1.92] | 1.174 | -0.251 [-2.03, 0.39] | +0.84 | 0.49 | 0.37 | 29% | 54% | 0.852 |
| `r2_blend_nnls_ergfix` | 28 | 1.510 [1.07, 1.92] | 1.149 | -0.272 [-1.97, 0.37] | +0.87 | 0.49 | 0.43 | 32% | 57% | 0.793 |
| `r2_svr_dm` | 28 | 1.511 [1.08, 1.93] | 1.192 | -0.274 [-2.19, 0.41] | +0.82 | 0.51 | 0.39 | 29% | 46% | 0.957 |
| `svr_desc` | 28 | 1.516 [1.07, 1.92] | 1.160 | -0.282 [-1.83, 0.30] | +0.80 | 0.43 | 0.39 | 29% | 57% | 0.899 |
| `r2_catboost_dm` | 28 | 1.524 [1.08, 1.97] | 1.177 | -0.296 [-2.21, 0.39] | +0.93 | 0.54 | 0.42 | 29% | 57% | 0.833 |
| `gnn_hybrid_s43` | 28 | 1.548 [1.04, 1.99] | 1.137 | -0.337 [-2.19, 0.37] | +0.74 | 0.43 | 0.41 | 39% | 61% | 0.922 |
| `svr_desc_v2` | 28 | 1.550 [1.10, 1.96] | 1.197 | -0.341 [-1.96, 0.25] | +0.81 | 0.41 | 0.37 | 29% | 61% | 0.893 |
| `ridge_desc` | 28 | 1.553 [1.14, 1.92] | 1.214 | -0.346 [-1.88, 0.23] | +0.91 | 0.41 | 0.22 | 29% | 50% | 1.088 |
| `gnn_hybrid` | 28 | 1.601 [1.07, 2.08] | 1.138 | -0.430 [-1.82, 0.19] | +0.79 | 0.37 | 0.44 | 32% | 64% | 0.929 |
| `gnn_hybrid_s44` | 28 | 1.623 [1.12, 2.08] | 1.244 | -0.470 [-2.36, 0.24] | +0.83 | 0.39 | 0.40 | 21% | 57% | 0.910 |
| `mlp_desc` | 28 | 1.665 [1.23, 2.05] | 1.337 | -0.547 [-2.42, 0.13] | +0.82 | 0.43 | 0.42 | 21% | 50% | 0.939 |
| `gnn_mpnn` | 28 | 1.722 [1.23, 2.17] | 1.331 | -0.655 [-2.44, 0.04] | +1.04 | 0.38 | 0.36 | 29% | 50% | 0.927 |
| `knn_morganbits_k5` | 28 | 1.854 [1.31, 2.33] | 1.388 | -0.918 [-2.88, -0.11] | +0.81 | 0.11 | 0.07 | 39% | 54% | 1.605 |
| `gnn_mpnn_s43` | 28 | 1.880 [1.32, 2.41] | 1.408 | -0.971 [-2.78, -0.19] | +1.09 | 0.15 | 0.13 | 29% | 50% | 0.955 |
