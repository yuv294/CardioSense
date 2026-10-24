"""Why does everything stop at about 0.72?  Two standard diagnostics, development rows only.

1. Input ablation: how much of the score does each group of inputs buy?
2. Learning curve: if we had more patients, would the score rise?  (If the held-out curve has
   flattened and sits close to the training curve, more data of the same kind will not help.)
Logistic regression, preprocessing inside every fold, 3 x repeated stratified 5-fold CV.
Writes reports/ceiling.json and reports/ceiling.png.
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from config import DATA_RAW, REPORTS, FEATURES, ENGINEERED_FEATURES, TARGET, RANDOM_STATE
from features import add_engineered_features, add_structure_features, STRUCTURE_FEATURES
from ladder import prep

warnings.filterwarnings("ignore")
raw = pd.read_csv(DATA_RAW)
yall = raw[TARGET].astype(int).values
idx, _ = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=RANDOM_STATE, stratify=yall)
dev = add_structure_features(add_engineered_features(raw.iloc[idx].reset_index(drop=True)))
y = yall[idx]
ALL31 = FEATURES + ENGINEERED_FEATURES + STRUCTURE_FEATURES
cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=11).split(dev, y))


def score(cols, C, frac=1.0):
    te_auc, tr_auc = [], []
    for k, (tr, te) in enumerate(cv):
        rng = np.random.default_rng(k)
        tr = rng.choice(tr, int(len(tr) * frac), replace=False)
        A, B = prep(dev.iloc[tr], dev.iloc[te], cols)
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(A, y[tr])
        te_auc.append(roc_auc_score(y[te], m.predict_proba(B)[:, 1]))
        tr_auc.append(roc_auc_score(y[tr], m.predict_proba(A)[:, 1]))
    return float(np.mean(te_auc)), float(np.mean(tr_auc))


subsets = [("Age only", ["age"]), ("Age + sex", ["age", "male"]), ("+ systolic BP", ["age", "male", "sysBP"]),
           ("+ smoking, diabetes", ["age", "male", "sysBP", "cigsPerDay", "diabetes"]),
           ("All 15 raw inputs", FEATURES), ("All 31 features", ALL31)]
abl = {n: score(c, 0.1)[0] for n, c in subsets}
sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
n_full = int(len(dev) * 0.8)
lc = {str(int(n_full * f)): score(ALL31, 0.03, f) for f in sizes}
json.dump({"ablation_auroc": abl, "learning_curve": {k: {"heldout": v[0], "train": v[1]} for k, v in lc.items()}},
          open(REPORTS / "ceiling.json", "w"), indent=2)
for n, v in abl.items():
    print(f"{n:24s} {v:.3f}")
for k, v in lc.items():
    print(f"train size {k:>5s}  held-out {v[0]:.3f}  train {v[1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
names = list(abl)
ax[0].barh(names[::-1], [abl[n] for n in names][::-1], color="#0E7C86")
ax[0].set_xlim(0.6, 0.76)
for i, n in enumerate(names[::-1]):
    ax[0].text(abl[n] + 0.002, i, f"{abl[n]:.3f}", va="center", fontsize=9)
ax[0].set_xlabel("Cross-validated AUROC")
ax[0].set_title("What each group of inputs buys")
ax[0].spines[["top", "right"]].set_visible(False)
xs = [int(k) for k in lc]
ax[1].plot(xs, [v[0] for v in lc.values()], "o-", color="#0E7C86", lw=2, label="Held-out patients")
ax[1].plot(xs, [v[1] for v in lc.values()], "o--", color="#8896A6", lw=1.6, label="Patients it trained on")
ax[1].set_xlabel("Number of training patients")
ax[1].set_ylabel("AUROC")
ax[1].set_title("Learning curve")
ax[1].legend(frameon=False, fontsize=9)
ax[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS / "ceiling.png", dpi=170)
