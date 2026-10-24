"""Locked-test check, paired: the served model against Framingham on the SAME 848 patients.
A paired bootstrap resamples patients once and scores both models on each resample, so the
interval for the DIFFERENCE is much tighter than comparing two separate intervals.
Writes reports/paired_test.json."""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED, ARTIFACTS, REPORTS
from baseline_frs import frs_risk_frame

X = np.load(DATA_PROCESSED / "X_test_proc.npy")
y = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].to_numpy()
frs = frs_risk_frame(pd.read_csv(DATA_PROCESSED / "X_test.csv"))
served = joblib.load(ARTIFACTS / "model_logreg.joblib").predict_proba(X)[:, 1]   # the ranking the served model is built on
rng = np.random.default_rng(0)
d = []
for _ in range(2000):
    b = rng.integers(0, len(y), len(y))
    if y[b].min() == y[b].max():
        continue
    d.append(roc_auc_score(y[b], served[b]) - roc_auc_score(y[b], frs[b]))
out = {"served_auroc": float(roc_auc_score(y, served)), "framingham_auroc": float(roc_auc_score(y, frs)),
       "difference": float(roc_auc_score(y, served) - roc_auc_score(y, frs)),
       "difference_ci": [float(v) for v in np.percentile(d, [2.5, 97.5])]}
(REPORTS / "paired_test.json").write_text(json.dumps(out, indent=2))
print(out)
