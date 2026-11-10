**Primary UI model: [`blend_nnls_family`](blend_nnls_family/)**, a blend of 8 models, with the
pre-registered final model [`blend_nnls`](blend_nnls/) (12 models) as an equally accurate
alternative.

* Accuracy: on the in-domain test set, `blend_nnls_family` scored RMSE 0.796 / R² 0.881 and
  `blend_nnls` scored RMSE 0.799 / R² 0.880. They are statistically indistinguishable: the
  paired-bootstrap ΔRMSE 95% CI is [-0.006, +0.001], and p = 0.90 that the family blend is at
  least as good. Both are significantly better than every single model (vs the best single model,
  `lgbm_desc_mordred`: ΔRMSE +0.034, CI [0.016, 0.051] for molecules and [0.012, 0.057] for clusters).
* Why the family blend for a UI: it needs fewer components (8 vs 12), so it has fewer files and
  loads faster, and it has one model per family, so it is less sensitive to any one component.
  **Transparency note:** `blend_nnls` was the pre-registered selection. The family blend is
  recommended for engineering reasons only, and its test score was not used to rank it above
  `blend_nnls`.
* Heads-up for a UI: both blends include `ridge_desc` at a small weight (about 0.02–0.03). Ridge
  extrapolates badly on out-of-domain inputs, so **always run
  [`applicability_domain.py`](applicability_domain.py) first** and show its flags. On the test
  set, `blend_nnls` scored RMSE 1.80 on the 48 out-of-domain entries (metal-containing or
  multi-component mixtures), against 0.80 in-domain.
* Show a confidence band from the nearest-neighbour Tanimoto similarity (to the training
  set) returned by `applicability_domain.check()`. Observed in-domain test RMSE for
  `blend_nnls`: 0.58 at similarity ≥ 0.6, 0.80 at 0.4–0.6, and 0.93 below 0.4.

**Lightweight fallback: [`lgbm_desc_mordred`](lgbm_desc_mordred/)**, a single LightGBM model
(RDKit + Mordred descriptors). In-domain test RMSE 0.832 / R² 0.870. It needs only
rdkit + mordredcommunity + lightgbm and has no neural-network or torch dependency. If you want
no Mordred either, use [`lgbm_desc`](lgbm_desc/) (RDKit descriptors only, test RMSE 0.840).

Not recommended for a UI: `ridge_desc`, `esol4_linear`, `knn_morganbits_k5` and `rf_morgan`
(clearly worse). The GNNs are useful only as blend components (on their own they score test
RMSE about 0.93 and need torch).

**Round 2 update (2026-10-02).** The round-2 pre-registered candidate, [`r2_blend_nnls`](r2_blend_nnls/), is an
NNLS blend of 18 models drawn from an enlarged pool that adds GNN seed replicates, CatBoost, Avalon/ErG and
Mordred variants. On the in-domain test set it scored RMSE 0.793 / R² 0.882, against 0.799 / 0.880 for
`blend_nnls`. The improvement is **not** confirmed by the pre-registered test: on the paired cluster
bootstrap ΔRMSE = −0.006, 95% CI [−0.014, +0.002], one-sided p = 0.079. The recommendation above is
therefore unchanged. If you want the round-2 blend, use its reproducible rebuild [`r2_blend_nnls_ergfix`](r2_blend_nnls_ergfix/) (test RMSE 0.793), not
`r2_blend_nnls`: two of the latter's components use a non-deterministic RDKit ErG fingerprint (see REPORT R2.6). Either needs 18 component
folders and mordred + torch. **Do not use `r2_lgbm_d_av_erg` or `r2_lgbm_desc_mordred_avalon_erg`** (reproducibility FAIL); use their `*_erg2` retrains. If a single GNN is wanted, prefer the seed-averaged
[`r2_gnn_hybrid_ens3`](r2_gnn_hybrid_ens3/) (test RMSE 0.888) over any single-seed GNN (0.91–0.93).
