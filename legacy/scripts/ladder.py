"""The improvement ladder: does each extra idea actually help?  (points 1, 2, 3, 5)

Each step adds ONE idea to the one before it. Every step is scored the same way:
  * development rows only (the 3,392 training patients; the locked test set is not touched)
  * 5 x repeated stratified 5-fold cross-validation (25 held-out folds)
  * the penalty C is chosen INSIDE each training fold (nested), by the one-standard-
    error rule, so the score is not flattered by tuning
  * preprocessing is fitted inside each fold
  * every step is compared with Framingham on the SAME patients (paired bootstrap)

Steps (in order of added complexity):
  S2  logistic regression, our original 23 features                  (the current served model)
  S3  + Framingham-style structure: logs, sex interactions, treated blood pressure   (point 2)
  S4  + "was missing" flags for glucose, education, BP medication, cholesterol       (point 5)
  S5b + the Framingham log-odds as one more feature                                  (point 1, free weight)
  S5a + the Framingham log-odds as an OFFSET (weight fixed at 1, only corrections learned) (point 1)
Baselines (never served): Framingham as published; Framingham recalibrated (intercept + slope).
For comparison only: shallow gradient boosting with and without the Framingham feature.

SELECTION RULE (written before the results were seen): among S2..S5a, take the first step in the
order above whose mean cross-validated average precision is within one standard error of the best
step. Only a step that clearly beats the simpler ones is allowed to replace S2.
"""
import json
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import DATA_RAW, DATA_PROCESSED, REPORTS, FEATURES, ENGINEERED_FEATURES, TARGET, RANDOM_STATE
from features import (add_engineered_features, add_structure_features, add_missing_flags,
                      add_frs_logit, STRUCTURE_FEATURES, MISSING_FLAGS)
from baseline_frs import frs_risk_frame
from offset_logreg import OffsetLogisticRegression

warnings.filterwarnings("ignore")
ALL23 = FEATURES + ENGINEERED_FEATURES
BINARY = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "education",
          "hyp_and_diab", "high_glucose", "male_x_smoker", "male_x_diabetes"] + MISSING_FLAGS
GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]          # ascending = simpler first
STEPS = {  # name: (columns, offset column or None, extra unscaled feature column or None)
    "S2": (ALL23, None),
    "S3": (ALL23 + STRUCTURE_FEATURES, None),
    "S4": (ALL23 + STRUCTURE_FEATURES + MISSING_FLAGS, None),
    "S5b": (ALL23 + STRUCTURE_FEATURES + MISSING_FLAGS + ["frs_logit"], None),
    "S5a": (ALL23 + STRUCTURE_FEATURES + MISSING_FLAGS, "frs_logit"),
}
ORDER = ["S2", "S3", "S4", "S5b", "S5a"]


def build_frame(raw):
    """Raw rows -> every column any step could need (flags first, before any imputation)."""
    d = add_missing_flags(raw)
    d = add_engineered_features(d)
    d = add_structure_features(d)
    return add_frs_logit(d)


def prep(tr, te, cols, offset=None):
    cont = [c for c in cols if c not in BINARY]
    bina = [c for c in cols if c in BINARY]
    ic = SimpleImputer(strategy="median").fit(tr[cont])
    sc = StandardScaler().fit(ic.transform(tr[cont]))
    ib = SimpleImputer(strategy="most_frequent").fit(tr[bina]) if bina else None
    io = SimpleImputer(strategy="median").fit(tr[[offset]]) if offset else None

    def f(d):
        parts = [sc.transform(ic.transform(d[cont]))]
        if bina:
            parts.append(ib.transform(d[bina]))
        if offset:
            parts.append(io.transform(d[[offset]]))
        return np.hstack(parts)
    return f(tr), f(te)


def pick_C_1se(make, A, y, seed):
    """Inner 5-fold CV; simplest C within one standard error of the best average precision."""
    sc = np.zeros((len(GRID), 5))
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(A, y)):
        for g, C in enumerate(GRID):
            m = make(C).fit(A[a], y[a])
            sc[g, k] = average_precision_score(y[b], m.predict_proba(A[b])[:, 1])
    mean, se = sc.mean(1), sc.std(1, ddof=1) / np.sqrt(5)
    best = int(np.argmax(mean))
    return next(g for g in range(len(GRID)) if mean[g] >= mean[best] - se[best])


def main():
    raw = pd.read_csv(DATA_RAW)
    y_all = raw[TARGET].astype(int).values
    # the same development rows as preprocess.py: the training portion of the locked split
    Xtr_raw = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    ytr = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].values.astype(int)
    # X_train.csv holds only the 23 columns; rebuild the full frame from the raw file by position
    from sklearn.model_selection import train_test_split
    idx_tr, _ = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=RANDOM_STATE, stratify=y_all)
    dev = build_frame(raw.iloc[idx_tr].reset_index(drop=True))
    y = y_all[idx_tr]
    assert (y == ytr).all(), "development rows do not match the saved training split"
    N = len(dev)

    names = ["Framingham", "Framingham recalibrated"] + ORDER + ["HGB (23 features)", "HGB (+ Framingham feature)"]
    R = 5
    oof = {k: np.zeros((R, N)) for k in names}
    fold_ap = {k: [] for k in names}
    chosen = {k: [] for k in ORDER}
    for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=7).split(dev, y)):
        r = i // 5
        dtr, dte, ytr_f, yte_f = dev.iloc[tr], dev.iloc[te], y[tr], y[te]
        preds = {}
        preds["Framingham"] = frs_risk_frame(dte[FEATURES].fillna(dtr[FEATURES].median()))
        # recalibrated Framingham: a 2-parameter logistic fit on the training fold
        a, b = prep(dtr, dte, ["frs_logit"])
        preds["Framingham recalibrated"] = LogisticRegression(C=1e6, max_iter=1000).fit(a, ytr_f).predict_proba(b)[:, 1]
        for s in ORDER:
            cols, off = STEPS[s]
            A, B = prep(dtr, dte, cols, off)
            make = (lambda C: OffsetLogisticRegression(C=C)) if off else \
                   (lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
            g = pick_C_1se(make, A, ytr_f, seed=100 + i)
            chosen[s].append(GRID[g])
            preds[s] = make(GRID[g]).fit(A, ytr_f).predict_proba(B)[:, 1]
        hp = dict(max_depth=2, learning_rate=0.03, max_iter=100, l2_regularization=5.0, min_samples_leaf=50,
                  early_stopping=False, random_state=0)
        A, B = prep(dtr, dte, ALL23)
        preds["HGB (23 features)"] = HistGradientBoostingClassifier(**hp).fit(A, ytr_f).predict_proba(B)[:, 1]
        A, B = prep(dtr, dte, STEPS["S5b"][0])
        preds["HGB (+ Framingham feature)"] = HistGradientBoostingClassifier(**hp).fit(A, ytr_f).predict_proba(B)[:, 1]
        for k, p in preds.items():
            oof[k][r, te] = p
            fold_ap[k].append(average_precision_score(yte_f, p))
        print(f"fold {i + 1}/25", flush=True)

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, N, N) for _ in range(1000)]
    mean_pred = {k: oof[k].mean(0) for k in names}

    def paired(k, ref):
        d = [roc_auc_score(y[b], mean_pred[k][b]) - roc_auc_score(y[b], mean_pred[ref][b]) for b in boots]
        return float(np.mean(d)), [float(v) for v in np.percentile(d, [2.5, 97.5])]

    out = {"n_dev": N, "events": int(y.sum()), "repeats": R, "folds": 25, "steps": {}}
    for k in names:
        aucs = [roc_auc_score(y, oof[k][r]) for r in range(R)]
        ap = np.array(fold_ap[k])
        e = {"auroc": float(np.mean(aucs)), "auroc_sd_over_repeats": float(np.std(aucs)),
             "ap": float(ap.mean()), "ap_se": float(ap.std(ddof=1) / np.sqrt(5))}
        if k != "Framingham":
            d, ci = paired(k, "Framingham")
            e["diff_vs_framingham"], e["diff_vs_framingham_ci"] = d, ci
        if k in ORDER and k != "S2":
            d, ci = paired(k, "S2")
            e["diff_vs_S2"], e["diff_vs_S2_ci"] = d, ci
        if k in ORDER:
            e["chosen_C_counts"] = {str(c): chosen[k].count(c) for c in GRID if c in chosen[k]}
        out["steps"][k] = e

    # selection rule (see docstring)
    ap = {s: out["steps"][s]["ap"] for s in ORDER}
    best = max(ORDER, key=lambda s: ap[s])
    margin = out["steps"][best]["ap_se"]
    selected = next(s for s in ORDER if ap[s] >= ap[best] - margin)
    out["best_step_by_ap"], out["one_se"], out["selected_step"] = best, margin, selected

    print(f"\n{'model':30s} AUROC   AP     diff vs Framingham [95% CI]     diff vs S2 [95% CI]")
    for k in names:
        e = out["steps"][k]
        a = f"{e.get('diff_vs_framingham', 0):+.3f} [{e['diff_vs_framingham_ci'][0]:+.3f},{e['diff_vs_framingham_ci'][1]:+.3f}]" if "diff_vs_framingham" in e else "-"
        b = f"{e['diff_vs_S2']:+.3f} [{e['diff_vs_S2_ci'][0]:+.3f},{e['diff_vs_S2_ci'][1]:+.3f}]" if "diff_vs_S2" in e else "-"
        print(f"{k:30s} {e['auroc']:.3f}  {e['ap']:.3f}  {a:30s}  {b}")
    print(f"\nbest step by AP: {best}  one SE: {margin:.4f}  -> selected by the rule: {selected}")
    (REPORTS / "ladder.json").write_text(json.dumps(out, indent=2))
    np.savez(REPORTS / "ladder_oof.npz", y=y, idx=idx_tr, **{k.replace(" ", "_"): v for k, v in oof.items()})


if __name__ == "__main__":
    main()
