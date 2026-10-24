"""Overfitting check and model facts, for the report and the viva.

For every model this prints and saves (reports/diagnostics.json):
    train AUROC   on the patients it learned from
    CV AUROC      5-fold cross-validation inside the training set
    test AUROC    on the locked test set
    gap           train minus test (a big gap means overfitting)
plus facts about each fitted model (weights, tree sizes, iterations).
This script only REPORTS; no model choice depends on it.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config import DATA_PROCESSED, ARTIFACTS, REPORTS, RANDOM_STATE
from baseline_frs import frs_risk_frame

Xtr = np.load(DATA_PROCESSED / "X_train_proc.npy")
Xte = np.load(DATA_PROCESSED / "X_test_proc.npy")
ytr = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].to_numpy()
yte = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].to_numpy()
cols = json.loads((ARTIFACTS / "processed_columns.json").read_text())
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

out = {"models": {}, "facts": {}}
for name in ["logreg", "rf", "hgb", "mlp", "stack"]:
    m = joblib.load(ARTIFACTS / f"model_{name}.joblib")
    tr = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])
    te = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    cva = (cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1).mean()
           if name != "stack" else None)
    out["models"][name] = {"train_auroc": round(tr, 3),
                           "cv_auroc": None if cva is None else round(float(cva), 3),
                           "test_auroc": round(te, 3), "gap": round(tr - te, 3)}
    print(f"{name:7s} train {tr:.3f}   cv {'n/a  ' if cva is None else f'{cva:.3f}'}   "
          f"test {te:.3f}   gap {tr - te:+.3f}", flush=True)

frs = {s: roc_auc_score(y, frs_risk_frame(pd.read_csv(DATA_PROCESSED / f"X_{s}.csv")))
       for s, y in (("train", ytr), ("test", yte))}
out["framingham"] = {k: round(float(v), 3) for k, v in frs.items()}
print(f"framingham (no fitting)  train {frs['train']:.3f}   test {frs['test']:.3f}")

lr = joblib.load(ARTIFACTS / "model_logreg.joblib")
out["facts"]["logreg"] = {"C": lr.C, "iterations": int(lr.n_iter_[0]),
    "top_weights": [(c, round(float(w), 3)) for c, w in
                    sorted(zip(cols, lr.coef_[0]), key=lambda t: -abs(t[1]))[:10]]}
rf = joblib.load(ARTIFACTS / "model_rf.joblib")
out["facts"]["rf"] = {"params": {k: v for k, v in rf.get_params().items()
                                 if k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features")},
    "mean_depth": round(float(np.mean([e.get_depth() for e in rf.estimators_])), 1),
    "mean_leaves": round(float(np.mean([e.get_n_leaves() for e in rf.estimators_])), 1),
    "top_importances": [(c, round(float(v), 3)) for c, v in
                        sorted(zip(cols, rf.feature_importances_), key=lambda t: -t[1])[:8]]}
h = joblib.load(ARTIFACTS / "model_hgb.joblib")
out["facts"]["hgb"] = {"rounds_run": int(h.n_iter_), "max_rounds": h.max_iter,
    "params": {k: v for k, v in h.get_params().items()
               if k in ("learning_rate", "max_depth", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")}}
mp = joblib.load(ARTIFACTS / "model_mlp.joblib")
out["facts"]["mlp"] = {"hidden": list(mp.hidden_layer_sizes) if isinstance(mp.hidden_layer_sizes, tuple) else [mp.hidden_layer_sizes],
    "alpha": mp.alpha, "epochs_run": int(mp.n_iter_), "max_iter": mp.max_iter,
    "n_weights": int(sum(c.size for c in mp.coefs_) + sum(i.size for i in mp.intercepts_))}
st = joblib.load(ARTIFACTS / "model_stack.joblib")
out["facts"]["stack"] = {"meta_weights": {n: round(float(w), 3) for (n, _), w in
                                          zip(st.estimators, st.final_estimator_.coef_[0])}}
(REPORTS / "diagnostics.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["facts"], indent=1)[:3000])
