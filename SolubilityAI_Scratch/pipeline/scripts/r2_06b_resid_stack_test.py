"""Round 2 exploratory (CV-only, nested): NNLS blend + LightGBM residual correction on meta features.
Result: worse than the plain NNLS blend (0.891 / 0.895 vs 0.885); not saved as a model."""
import sys; sys.argv=["x"]
src=open('scripts/r2_06_stack.py').read().split("all_rows = np.arange")[0]
__file__='/home/claude/sol/scripts/r2_06_stack.py'
exec(src)
import lightgbm as lgb
for params in [dict(n_estimators=100, learning_rate=0.03, num_leaves=7, min_child_samples=100),
               dict(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=200, reg_lambda=10)]:
    oof=np.zeros(len(y))
    for k in range(5):
        tr,va=np.where(folds!=k)[0],np.where(folds==k)[0]
        # inner nested blend for training rows: blend OOF within the 4 folds
        b=np.zeros(len(tr)); ftr=folds[tr]
        for j in np.unique(ftr):
            a,c=tr[ftr!=j],tr[ftr==j]; w,i0=fit_nnls(P[a],y[a]); b[ftr==j]=P[c]@w+i0
        w,i0=fit_nnls(P[tr],y[tr]); bva=P[va]@w+i0
        m=lgb.LGBMRegressor(**params,subsample=0.8,subsample_freq=1,random_state=42,verbose=-1,n_jobs=2).fit(np.column_stack([b,META[tr]]),y[tr]-b)
        oof[va]=bva+m.predict(np.column_stack([bva,META[va]]))
    print(params, runner.metrics(y,oof))
