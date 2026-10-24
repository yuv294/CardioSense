"""Generate evaluation figures for the report (ROC, PR, calibration, importance)."""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

from config import DATA_PROCESSED, ARTIFACTS, REPORTS
from baseline_frs import frs_risk_frame

y_te = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].to_numpy()
X_te = np.load(DATA_PROCESSED / "X_test_proc.npy")
served = joblib.load(ARTIFACTS / "model_served.joblib")
p = served.predict_proba(X_te)[:, 1]
frs = frs_risk_frame(pd.read_csv(DATA_PROCESSED / "X_test.csv"))

# ROC
plt.figure(figsize=(5, 5))
for label, pr in [("CardioSense (calibrated LR)", p), ("Framingham baseline", frs)]:
    fpr, tpr, _ = roc_curve(y_te, pr)
    plt.plot(fpr, tpr, label=label)
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve"); plt.legend(); plt.tight_layout()
plt.savefig(REPORTS / "roc_curve.png", dpi=130); plt.close()

# PR
plt.figure(figsize=(5, 5))
for label, pr in [("CardioSense", p), ("Framingham", frs)]:
    prec, rec, _ = precision_recall_curve(y_te, pr)
    plt.plot(rec, prec, label=label)
plt.axhline(y_te.mean(), ls="--", c="k", alpha=0.3, label=f"base rate {y_te.mean():.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curve"); plt.legend(); plt.tight_layout()
plt.savefig(REPORTS / "pr_curve.png", dpi=130); plt.close()

# Calibration
rep = json.loads((REPORTS / "evaluation.json").read_text())
cc = rep["calibration_curve"]
plt.figure(figsize=(5, 5))
plt.plot(cc["mean_predicted"], cc["fraction_positive"], "o-", label="CardioSense")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
plt.xlabel("Predicted risk"); plt.ylabel("Observed frequency")
plt.title("Calibration"); plt.legend(); plt.tight_layout()
plt.savefig(REPORTS / "calibration.png", dpi=130); plt.close()

# Feature influence
exp = json.loads((ARTIFACTS / "explainer.json").read_text())
items = sorted(exp["coef"].items(), key=lambda kv: kv[1])
names = [k for k, _ in items]; vals = [v for _, v in items]
plt.figure(figsize=(6.5, 7))
plt.barh(names, vals)
plt.xlabel("Coefficient (standardised)"); plt.title("Feature influence (served model)")
plt.tight_layout(); plt.savefig(REPORTS / "feature_influence.png", dpi=130); plt.close()

# Model comparison with bootstrap CIs
NAMES = {"framingham_baseline": "Framingham score", "logreg": "Logistic regression",
         "stack": "Stacked ensemble", "mlp": "Neural network", "rf": "Random forest",
         "hgb": "Gradient boosting"}
rows = sorted(rep["metrics"], key=lambda r: r["auroc"])
plt.figure(figsize=(6.5, 3.8))
ys = range(len(rows))
plt.barh(list(ys), [r["auroc"] for r in rows],
         xerr=[[r["auroc"] - r["auroc_ci_low"] for r in rows],
               [r["auroc_ci_high"] - r["auroc"] for r in rows]],
         color=["#E0A030" if r["model"] == "framingham_baseline" else "#0E7C86" if r["model"] == "logreg" else "#8896A6" for r in rows],
         capsize=3)
plt.yticks(list(ys), [NAMES.get(r["model"], r["model"]) for r in rows])
plt.xlim(0.5, 0.8); plt.xlabel("AUROC on held-out test set (95% bootstrap CI)")
plt.title("Model comparison"); plt.tight_layout()
plt.savefig(REPORTS / "model_comparison.png", dpi=150); plt.close()

# Confusion matrix at the operating threshold
c = rep["confusion_at_threshold"]
cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
            color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Predicted clear", "Predicted flagged"])
ax.set_yticks([0, 1], ["No event", "Had event"])
ax.set_title(f"Outcomes at {rep['operating_threshold']*100:.0f}% threshold")
plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix.png", dpi=150); plt.close()

print("Saved figures to reports/")
