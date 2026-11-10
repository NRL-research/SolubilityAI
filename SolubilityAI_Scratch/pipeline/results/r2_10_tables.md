### Table R2-A. Round-2 configurations (plus the round-1 reference `blend_nnls`), sorted by in-domain CV RMSE

CV = 5-fold grouped CV on the 7,108 in-domain training molecules (± SD of per-fold RMSE on the model's own training set). Test = one-time evaluation on the 1,804 in-domain test molecules [95% cluster-bootstrap CI]. Δ = test RMSE minus blend_nnls's on the same resamples; p = share of paired cluster-bootstrap resamples in which the model is NOT better than blend_nnls.

| config | features | CV RMSE ± SD | CV R² | Test RMSE [95% CI] | Test MAE | Test R² [95% CI] | Δ vs blend_nnls [95% CI] | p | Full-test RMSE / R² |
|---|---|---|---|---|---|---|---|---|---|
| `r2_blend_nnls` | (components) | 0.883 ± 0.026 | 0.859 | 0.793 [0.720, 0.877] | 0.540 | 0.882 [0.828, 0.917] | -0.006 [-0.014, +0.002] | 0.079 | 0.831 / 0.871 |
| `r2_blend_nnls_ergfix` | (components) | 0.883 ± 0.026 | 0.859 | 0.793 [0.720, 0.876] | 0.540 | 0.882 [0.828, 0.917] | -0.006 [-0.014, +0.002] | 0.083 | 0.831 / 0.871 |
| `blend_nnls` | (components) | 0.887 ± 0.023 | 0.858 | 0.799 [0.723, 0.884] | 0.546 | 0.880 [0.826, 0.916] | +0.000 [+0.000, +0.000] | 1.000 | 0.840 / 0.868 |
| `r2_stack_ridge` | (components) | 0.892 ± 0.025 | 0.856 | 0.798 [0.723, 0.886] | 0.548 | 0.880 [0.826, 0.915] | -0.000 [-0.010, +0.010] | 0.484 | 0.836 / 0.869 |
| `r2_stack_lgbm_meta` | (components) | 0.914 ± 0.024 | 0.849 | 0.814 [0.740, 0.896] | 0.565 | 0.875 [0.821, 0.911] | +0.016 [-0.006, +0.035] | 0.918 | 0.859 / 0.862 |
| `r2_lgbm_desc_mordred_avalon_erg2` | r2:desc+mordred+avalon+erg2 | 0.924 ± 0.028 | 0.846 | 0.819 [0.745, 0.900] | 0.574 | 0.874 [0.819, 0.911] | +0.020 [-0.001, +0.041] | 0.972 | 0.858 / 0.862 |
| `r2_lgbm_dm_huber` | desc+mordred | 0.924 ± 0.038 | 0.846 | 0.834 [0.754, 0.924] | 0.586 | 0.869 [0.809, 0.909] | +0.035 [+0.014, +0.057] | 1.000 | 0.870 / 0.858 |
| `r2_lgbm_desc_mordred_avalon_erg` | r2:desc+mordred+avalon+erg | 0.924 ± 0.032 | 0.846 | 0.822 [0.745, 0.904] | 0.574 | 0.873 [0.818, 0.911] | +0.023 [+0.003, +0.044] | 0.988 | 0.865 / 0.860 |
| `r2_catboost_dm` | desc+mordred | 0.925 ± 0.043 | 0.846 | 0.833 [0.751, 0.925] | 0.585 | 0.870 [0.809, 0.909] | +0.034 [+0.010, +0.059] | 0.998 | 0.876 / 0.856 |
| `r2_lgbm_dm_m2v` | r2:desc+mordred+mol2vec | 0.927 ± 0.034 | 0.845 | 0.834 [0.757, 0.924] | 0.590 | 0.869 [0.810, 0.908] | +0.036 [+0.011, +0.063] | 0.999 | 0.879 / 0.855 |
| `r2_lgbm_dm_crip_pack` | r2:desc+mordred+crippen_types+packing | 0.927 ± 0.032 | 0.845 | 0.835 [0.758, 0.922] | 0.585 | 0.869 [0.810, 0.908] | +0.036 [+0.015, +0.058] | 0.999 | 0.876 / 0.856 |
| `r2_lgbm_d_av_erg` | r2:desc+avalon+erg | 0.929 ± 0.023 | 0.844 | 0.822 [0.749, 0.902] | 0.567 | 0.873 [0.818, 0.910] | +0.023 [-0.004, +0.051] | 0.943 | 0.869 / 0.859 |
| `r2_lgbm_d_av_erg2` | r2:desc+avalon+erg2 | 0.930 ± 0.023 | 0.844 | 0.822 [0.749, 0.902] | 0.568 | 0.873 [0.819, 0.911] | +0.023 [-0.004, +0.049] | 0.952 | 0.872 / 0.858 |
| `r2_lgbm_d_ap_tt` | r2:desc+atompair+torsion | 0.940 ± 0.035 | 0.840 | 0.842 [0.762, 0.931] | 0.583 | 0.867 [0.806, 0.907] | +0.043 [+0.015, +0.072] | 0.998 | 0.886 / 0.853 |
| `r2_lgbm_d_crip_pack` | r2:desc+crippen_types+packing | 0.946 ± 0.036 | 0.838 | 0.837 [0.759, 0.923] | 0.587 | 0.868 [0.810, 0.907] | +0.039 [+0.016, +0.064] | 1.000 | 0.894 / 0.850 |
| `r2_gnn_hybrid_ens3` | graph+rdkit_desc(quantile-normal) | 0.950 ± 0.042 | 0.837 | 0.888 [0.803, 0.985] | 0.603 | 0.852 [0.784, 0.896] | +0.089 [+0.056, +0.125] | 1.000 | 0.919 / 0.842 |
| `r2_gnn_hybrid_ens2` | graph+rdkit_desc(quantile-normal) | 0.959 ± 0.043 | 0.834 | 0.902 [0.813, 0.999] | 0.611 | 0.847 [0.777, 0.892] | +0.103 [+0.067, +0.142] | 1.000 | 0.934 / 0.837 |
| `gnn_hybrid_s43` | graph+rdkit_desc(quantile-normal) | 0.974 ± 0.048 | 0.829 | 0.922 [0.835, 1.018] | 0.629 | 0.840 [0.768, 0.888] | +0.123 [+0.087, +0.164] | 1.000 | 0.954 / 0.830 |
| `gnn_hybrid_s44` | graph+rdkit_desc(quantile-normal) | 0.982 ± 0.040 | 0.826 | 0.910 [0.823, 1.009] | 0.629 | 0.845 [0.775, 0.889] | +0.111 [+0.075, +0.147] | 1.000 | 0.943 / 0.833 |
| `r2_gnn_mpnn_ens2` | graph | 0.988 ± 0.055 | 0.824 | 0.910 [0.821, 1.011] | 0.616 | 0.845 [0.771, 0.891] | +0.111 [+0.067, +0.164] | 1.000 | 0.955 / 0.829 |
| `gnn_mpnn_s43` | graph | 1.019 ± 0.050 | 0.813 | 0.955 [0.855, 1.077] | 0.647 | 0.829 [0.742, 0.881] | +0.156 [+0.096, +0.239] | 1.000 | 0.996 / 0.814 |
| `r2_svr_dm` | desc+mordred | 1.033 ± 0.035 | 0.807 | 0.957 [0.861, 1.052] | 0.653 | 0.828 [0.753, 0.880] | +0.158 [+0.104, +0.219] | 1.000 | 1.004 / 0.811 |
| `r2_svr_m2v` | r2:mol2vec | 1.083 ± 0.055 | 0.788 | 0.972 [0.881, 1.077] | 0.682 | 0.822 [0.746, 0.876] | +0.174 [+0.130, +0.222] | 1.000 | 1.056 / 0.791 |
| `r2_lgbm_m2v` | r2:mol2vec | 1.112 ± 0.051 | 0.777 | 1.038 [0.949, 1.134] | 0.755 | 0.798 [0.710, 0.859] | +0.239 [+0.195, +0.287] | 1.000 | 1.095 / 0.775 |

### Table R2-B. CV-only exploratory variants (not refit or saved)

| variant | in-domain nested CV RMSE | CV R² | note |
|---|---|---|---|
| nnls+lgbm_residual_A | 0.8913 | 0.8566 | 100 trees lr0.03 leaves7 min_child100 |
| nnls+lgbm_residual_B | 0.8954 | 0.8553 | 300 trees lr0.02 leaves7 min_child200 lambda10 |
| gated_nnls_by_nn_sim | 0.8862 | 0.8583 | separate NNLS weights for nn_sim <0.4 / 0.4-0.6 / >=0.6 |
| gated_nnls_by_MolWt | 0.8888 | 0.8574 | separate NNLS weights for MolWt <200 / 200-350 / >=350 |
