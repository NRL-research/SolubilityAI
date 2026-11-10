### Table A. All configurations: cross-validation and one-time test results

CV = pooled out-of-fold predictions, 5-fold grouped CV. 'own' = on the model's own training set; 'in-domain' = on the 7,108 in-domain training molecules (same folds for all models, so this column is comparable). Fold SD = standard deviation of per-fold RMSE. Test = in-domain test (n=1804), with 95% CI from a cluster bootstrap. p = paired bootstrap (molecule / cluster level) probability that the model is at least as good as the pre-registered `blend_nnls`.

| config | family | features | train set | CV RMSE own (fold SD) | CV RMSE in-domain | CV R² in-domain | Test RMSE [95% CI] | Test R² | Full-test RMSE / R² | ΔRMSE vs best [95% CI, molecule] | p (mol / cluster) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `blend_nnls` | blend | (components) | v1+v2 | 0.887 (0.023) | 0.887 | 0.858 | 0.799 [0.723, 0.884] | 0.880 | 0.840 / 0.868 | +0.000 [+0.000, +0.000] | — |
| `blend_nnls_family` | blend | (components) | v1+v2 | 0.888 (0.024) | 0.888 | 0.858 | 0.796 [0.723, 0.879] | 0.881 | 0.839 / 0.868 | -0.002 [-0.006, +0.001] | 0.8965 / 0.8470 |
| `blend_mean_top3` | blend | (components) | v1+v2 | 0.915 (0.026) | 0.915 | 0.849 | 0.823 [0.745, 0.910] | 0.873 | 0.860 / 0.862 | +0.024 [+0.009, +0.039] | 0.0005 / 0.0075 |
| `lgbm_desc_mordred` | lgbm | desc+mordred | v1 | 0.979 (0.036) | 0.923 | 0.846 | 0.832 [0.756, 0.917] | 0.870 | 0.869 / 0.859 | +0.034 [+0.016, +0.051] | 0.0000 / 0.0020 |
| `lgbm_mordred` | lgbm | mordred | v1 | 0.986 (0.029) | 0.931 | 0.844 | 0.845 [0.762, 0.939] | 0.866 | 0.883 / 0.854 | +0.047 [+0.027, +0.064] | 0.0000 / 0.0000 |
| `lgbm_desc_maccs` | lgbm | desc+maccs | v1 | 0.992 (0.033) | 0.935 | 0.842 | 0.852 [0.773, 0.936] | 0.864 | 0.892 / 0.851 | +0.053 [+0.032, +0.073] | 0.0000 / 0.0000 |
| `lgbm_desc_morgan` | lgbm | desc+morgan_counts_r2_2048 | v1 | 0.994 (0.034) | 0.936 | 0.842 | 0.839 [0.764, 0.921] | 0.868 | 0.891 / 0.851 | +0.040 [+0.021, +0.058] | 0.0000 / 0.0015 |
| `lgbm_desc_morgan_v2` | lgbm | desc+morgan_counts_r2_2048 | v2 | 0.937 (0.034) | 0.937 | 0.842 | 0.824 [0.753, 0.904] | 0.872 | 0.907 / 0.846 | +0.026 [+0.007, +0.044] | 0.0040 / 0.0255 |
| `lgbm_desc_v2` | lgbm | rdkit_desc | v2 | 0.940 (0.031) | 0.940 | 0.840 | 0.847 [0.768, 0.933] | 0.865 | 0.931 / 0.838 | +0.048 [+0.025, +0.071] | 0.0000 / 0.0005 |
| `catboost_desc_v2` | catboost | rdkit_desc | v2 | 0.941 (0.035) | 0.941 | 0.840 | 0.841 [0.763, 0.930] | 0.867 | 0.932 / 0.837 | +0.042 [+0.018, +0.066] | 0.0000 / 0.0030 |
| `lgbm_desc` | lgbm | rdkit_desc | v1 | 1.002 (0.031) | 0.941 | 0.840 | 0.840 [0.762, 0.928] | 0.867 | 0.891 / 0.851 | +0.041 [+0.019, +0.064] | 0.0005 / 0.0025 |
| `xgb_desc_morgan_v2` | xgb | desc+morgan_counts_r2_2048 | v2 | 0.941 (0.030) | 0.941 | 0.840 | 0.844 [0.768, 0.928] | 0.866 | 0.924 / 0.840 | +0.045 [+0.022, +0.069] | 0.0000 / 0.0000 |
| `catboost_desc` | catboost | rdkit_desc | v1 | 1.002 (0.039) | 0.943 | 0.839 | 0.846 [0.768, 0.936] | 0.866 | 0.898 / 0.849 | +0.047 [+0.022, +0.071] | 0.0000 / 0.0005 |
| `xgb_desc_morgan` | xgb | desc+morgan_counts_r2_2048 | v1 | 1.006 (0.035) | 0.944 | 0.839 | 0.851 [0.777, 0.932] | 0.864 | 0.906 / 0.846 | +0.052 [+0.031, +0.073] | 0.0000 / 0.0000 |
| `gnn_hybrid` | gnn_hybrid | graph+rdkit_desc(quantile-normal) | v1 | 1.051 (0.039) | 0.986 | 0.824 | 0.929 [0.838, 1.030] | 0.838 | 0.967 / 0.825 | +0.130 [+0.098, +0.164] | 0.0000 / 0.0000 |
| `et_desc_morgan` | et | desc+morgan_counts_r2_2048 | v1 | 1.047 (0.031) | 0.987 | 0.824 | 0.894 [0.811, 0.984] | 0.850 | 0.957 / 0.829 | +0.095 [+0.067, +0.123] | 0.0000 / 0.0000 |
| `svr_desc_v2` | svr | rdkit_desc | v2 | 0.990 (0.032) | 0.990 | 0.823 | 0.893 [0.813, 0.980] | 0.850 | 0.959 / 0.828 | +0.094 [+0.065, +0.126] | 0.0000 / 0.0000 |
| `svr_desc` | svr | rdkit_desc | v1 | 1.067 (0.042) | 0.990 | 0.823 | 0.899 [0.815, 0.991] | 0.848 | 0.972 / 0.823 | +0.100 [+0.074, +0.128] | 0.0000 / 0.0000 |
| `rf_desc_morgan` | rf | desc+morgan_counts_r2_2048 | v1 | 1.067 (0.033) | 1.006 | 0.817 | 0.918 [0.837, 1.008] | 0.842 | 0.984 / 0.819 | +0.119 [+0.090, +0.148] | 0.0000 / 0.0000 |
| `rf_desc_v2` | rf | rdkit_desc | v2 | 1.009 (0.032) | 1.009 | 0.816 | 0.920 [0.840, 1.010] | 0.841 | 1.014 / 0.807 | +0.122 [+0.089, +0.153] | 0.0000 / 0.0000 |
| `rf_desc` | rf | rdkit_desc | v1 | 1.069 (0.033) | 1.009 | 0.816 | 0.928 [0.845, 1.020] | 0.838 | 0.985 / 0.818 | +0.129 [+0.099, +0.159] | 0.0000 / 0.0000 |
| `gnn_mpnn` | gnn | graph | v1 | 1.099 (0.053) | 1.020 | 0.812 | 0.927 [0.838, 1.023] | 0.839 | 0.980 / 0.820 | +0.128 [+0.097, +0.158] | 0.0000 / 0.0000 |
| `mlp_desc` | mlp | rdkit_desc | v1 | 1.157 (0.047) | 1.062 | 0.796 | 0.939 [0.861, 1.022] | 0.834 | 1.007 / 0.810 | +0.140 [+0.110, +0.171] | 0.0000 / 0.0000 |
| `ridge_desc` | ridge | rdkit_desc | v1 | 2.235 (1.334) | 1.183 | 0.747 | 1.088 [0.997, 1.186] | 0.778 | 1.204 / 0.729 | +0.289 [+0.251, +0.326] | 0.0000 / 0.0000 |
| `rf_morgan` | rf | morgan_counts_r2_2048 | v1 | 1.294 (0.068) | 1.251 | 0.718 | 1.184 [1.088, 1.293] | 0.737 | 1.249 / 0.708 | +0.385 [+0.339, +0.433] | 0.0000 / 0.0000 |
| `esol4_linear` | linear | esol4 | v1 | 1.710 (0.179) | 1.537 | 0.574 | 1.462 [1.343, 1.571] | 0.598 | 1.588 / 0.528 | +0.664 [+0.610, +0.719] | 0.0000 / 0.0000 |
| `knn_morganbits_k5` | knn | morgan_bits_r2_2048 | v1 | 1.721 (0.089) | 1.682 | 0.490 | 1.605 [1.462, 1.761] | 0.516 | 1.639 / 0.497 | +0.807 [+0.737, +0.878] | 0.0000 / 0.0000 |

### Table B. Random hyperparameter searches (CV RMSE on the model's own training set)

| search | trials | best CV RMSE | worst CV RMSE |
|---|---|---|---|
| lgbm_desc | 14 | 1.002 | 1.049 |
| lgbm_desc_morgan | 8 | 0.994 | 1.003 |
| mlp_desc | 6 | 1.157 | 3.067 |
| svr_desc | 10 | 1.067 | 1.177 |
| xgb_desc_morgan | 6 | 1.006 | 1.024 |

### Table C. Subgroups on the test set, pre-registered model `blend_nnls`

| subgroup | n | RMSE | MAE | R² |
|---|---|---|---|---|
| all_in_domain | 1804 | 0.799 | 0.546 | 0.880 |
| out_of_domain(metal/mixture) | 48 | 1.801 | 1.433 | 0.411 |
| acyclic | 489 | 0.698 | 0.458 | 0.862 |
| ring | 1315 | 0.833 | 0.578 | 0.860 |
| salt/multicomponent | 113 | 0.981 | 0.732 | 0.488 |
| single_component | 1691 | 0.785 | 0.533 | 0.882 |
| nn_sim<0.4 | 557 | 0.926 | 0.667 | 0.798 |
| 0.4<=nn_sim<0.6 | 828 | 0.803 | 0.540 | 0.871 |
| nn_sim>=0.6 | 419 | 0.576 | 0.397 | 0.949 |
| logS<-6 | 199 | 1.063 | 0.721 | -0.293 |
| -6<=logS<-2 | 893 | 0.750 | 0.524 | 0.460 |
| logS>=-2 | 712 | 0.772 | 0.524 | 0.331 |

### Table D. External SC sets (read the caveats in the report)

| model | SC2_tight RMSE / R² (n=100) | SC2_loose RMSE / R² (n=32) | SC1 RMSE / R² (n=28) |
|---|---|---|---|
| `blend_nnls` | 0.98 / 0.40 | 1.68 / 0.38 | 1.47 / -0.21 |
| `blend_nnls_family` | 0.99 / 0.38 | 1.73 / 0.35 | 1.48 / -0.23 |
| `lgbm_desc_mordred` | 1.02 / 0.35 | 1.73 / 0.34 | 1.47 / -0.21 |
| `lgbm_desc` | 1.06 / 0.30 | 1.49 / 0.52 | 1.46 / -0.19 |
| `gnn_hybrid` | 1.18 / 0.13 | 2.06 / 0.07 | 1.60 / -0.43 |
| `svr_desc` | 0.99 / 0.39 | 1.89 / 0.22 | 1.52 / -0.28 |
| `esol4_linear` | 1.06 / 0.30 | 1.64 / 0.41 | 1.41 / -0.12 |

### Table E. Blend weights (fitted by NNLS on the in-domain training OOF predictions)

| component | `blend_nnls` weight | `blend_nnls_family` weight |
|---|---|---|
| `lgbm_mordred` | 0.2044 | 0.0000 |
| `gnn_mpnn` | 0.1915 | 0.1915 |
| `lgbm_desc_mordred` | 0.1838 | 0.3853 |
| `gnn_hybrid` | 0.1422 | 0.1549 |
| `lgbm_desc_maccs` | 0.0991 | 0.0000 |
| `svr_desc` | 0.0608 | 0.0000 |
| `svr_desc_v2` | 0.0353 | 0.0966 |
| `xgb_desc_morgan_v2` | 0.0345 | 0.1158 |
| `mlp_desc` | 0.0337 | 0.0431 |
| `ridge_desc` | 0.0189 | 0.0284 |
| `lgbm_desc_v2` | 0.0128 | 0.0000 |
| `knn_morganbits_k5` | 0.0006 | 0.0018 |
| intercept | 0.0412 | 0.0381 |
