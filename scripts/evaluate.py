"""Step 4: open the locked test set ONCE and judge everything on it.

Produces the numbers that go in the report:
  - AUROC and AUPRC for the 4 ML models AND the Framingham baseline
  - Brier score (calibration quality) for each
  - 95% bootstrap confidence intervals on AUROC (so we don't over-read a
    0.01 gap as a real difference)
  - a calibration curve for the best model, saved as JSON for the frontend
  - a chosen operating threshold, picked on the TRAINING set via CV-style
    out-of-fold predictions, never on the test set
  - confusion matrix at that threshold

It also calibrates the model chosen on cross-validation (isotonic) and saves it
as the model the API will actually serve. No test-set number is used to choose it.

Nothing here feeds back into training. This script only reads the locked set.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from config import DATA_PROCESSED, ARTIFACTS, REPORTS, RANDOM_STATE
from baseline_frs import frs_risk_frame

MODELS = ["logreg", "rf", "hgb", "mlp", "stack"]
RNG = np.random.default_rng(RANDOM_STATE)


def bootstrap_auroc_ci(y, p, n=2000):
    """95% percentile bootstrap CI for AUROC."""
    idx = np.arange(len(y))
    scores = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        if y[s].sum() == 0 or y[s].sum() == len(s):
            continue
        scores.append(roc_auc_score(y[s], p[s]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    X_tr = np.load(DATA_PROCESSED / "X_train_proc.npy")
    X_te = np.load(DATA_PROCESSED / "X_test_proc.npy")
    y_tr = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].to_numpy()
    y_te = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0].to_numpy()

    rows = []
    probs = {}

    # ----- ML models -----
    for name in MODELS:
        m = joblib.load(ARTIFACTS / f"model_{name}.joblib")
        p = m.predict_proba(X_te)[:, 1]
        probs[name] = p
        lo, hi = bootstrap_auroc_ci(y_te, p)
        rows.append({
            "model": name,
            "auroc": roc_auc_score(y_te, p),
            "auroc_ci_low": lo, "auroc_ci_high": hi,
            "auprc": average_precision_score(y_te, p),
            "brier": brier_score_loss(y_te, p),
        })

    # ----- Framingham rule-based baseline -----
    X_te_raw = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    frs = frs_risk_frame(X_te_raw)
    probs["framingham"] = frs
    lo, hi = bootstrap_auroc_ci(y_te, frs)
    rows.append({
        "model": "framingham_baseline",
        "auroc": roc_auc_score(y_te, frs),
        "auroc_ci_low": lo, "auroc_ci_high": hi,
        "auprc": average_precision_score(y_te, frs),
        "brier": brier_score_loss(y_te, frs),
    })

    results = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    print("\n=== Test-set performance (locked set, n={}) ===".format(len(y_te)))
    print(results.to_string(index=False,
          formatters={c: "{:.4f}".format for c in
                      ["auroc","auroc_ci_low","auroc_ci_high","auprc","brier"]}))

    # ----- choose the model to serve using TRAINING information only -----
    # The test set must not influence any decision, so the served model is picked
    # from the cross-validated average precision recorded in train.py.
    # The same one-standard-error rule used inside train.py is applied across
    # model families: take the SIMPLEST model whose CV score is within one
    # standard error of the best. The app also needs exact per-patient
    # explanations, which only a linear model gives, so a non-linear model must
    # beat the linear one by MORE than that margin to be served.
    cv_scores = json.loads((ARTIFACTS / "cv_results.json").read_text())
    top_name = max(cv_scores, key=lambda m: cv_scores[m]["cv_average_precision"])
    top = cv_scores[top_name]["cv_average_precision"]
    # The stack has no standard error of its own (it is not tuned by a grid), so when it is the
    # best model we use the average standard error of the tuned base models.
    ses = [v["one_se"] for v in cv_scores.values() if "one_se" in v]
    margin = cv_scores[top_name].get("one_se", float(np.mean(ses)) if ses else 0.005)
    simplicity = ["logreg", "hgb", "rf", "mlp", "stack"]
    best_ml = next(m for m in simplicity
                   if cv_scores[m]["cv_average_precision"] >= top - margin)
    print(f"\nBest CV model: {top_name} ({top:.4f}); one-SE margin {margin:.4f}")
    print(f"Served model chosen on CV (not on test): {best_ml}")
    base = joblib.load(ARTIFACTS / f"model_{best_ml}.joblib")

    # The app shows a probability to a person, so the served model is always
    # isotonic-calibrated (5-fold, TRAIN only). This is decided in advance; the
    # test Brier scores below are reported, not used to choose.
    calib = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calib.fit(X_tr, y_tr)
    p_cal = calib.predict_proba(X_te)[:, 1]
    brier_base = brier_score_loss(y_te, probs[best_ml])
    brier_cal = brier_score_loss(y_te, p_cal)
    print(f"  Brier uncalibrated: {brier_base:.4f}   calibrated: {brier_cal:.4f}")
    served, served_p, served_tag = calib, p_cal, "calibrated"
    joblib.dump(served, ARTIFACTS / "model_served.joblib")
    print(f"  Serving the {served_tag} {best_ml}.")

    # ----- choose operating threshold on TRAIN (out-of-fold), not on test ----
    # Critical: the threshold must be chosen on the SAME probability scale the
    # API serves. We serve `served` (possibly calibrated), so we derive the
    # out-of-fold probabilities from a clone of `served`, not of `base` --
    # otherwise calibration shifts the scale and the threshold means nothing.
    from sklearn.base import clone
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(served), X_tr, y_tr, cv=cv,
                            method="predict_proba")[:, 1]
    # target: catch >=70% of future cases (sensitivity), accept the alarms that implies
    target_sens = 0.70
    thr_grid = np.linspace(0.01, 0.9, 400)
    best_thr = 0.5
    for t in thr_grid:
        if recall_score(y_tr, (oof >= t).astype(int)) >= target_sens:
            best_thr = t
    print(f"\nThreshold picked on TRAIN for >= {target_sens:.0%} sensitivity: {best_thr:.3f}")

    # apply that threshold to the test set
    yhat = (served_p >= best_thr).astype(int)
    cm = confusion_matrix(y_te, yhat)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix on test set at this threshold:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"  sensitivity/recall: {recall_score(y_te, yhat):.3f}")
    print(f"  precision:          {precision_score(y_te, yhat):.3f}")
    print(f"  specificity:        {tn/(tn+fp):.3f}")
    print(f"  F1:                 {f1_score(y_te, yhat):.3f}")

    # ----- calibration curve for the frontend -----
    frac_pos, mean_pred = calibration_curve(y_te, served_p, n_bins=10, strategy="quantile")
    calib_json = {"mean_predicted": mean_pred.tolist(),
                  "fraction_positive": frac_pos.tolist()}

    # ----- save report bundle -----
    served_ci = bootstrap_auroc_ci(y_te, served_p)
    report = {
        "n_test": int(len(y_te)),
        "test_positive_rate": float(y_te.mean()),
        "metrics": results.to_dict(orient="records"),
        "served_model": {"base": best_ml, "variant": served_tag},
        # the calibrated model the app actually serves (the row in "metrics" is the raw base model)
        "served_metrics": {"auroc": float(roc_auc_score(y_te, served_p)),
                           "auroc_ci_low": served_ci[0], "auroc_ci_high": served_ci[1],
                           "auprc": float(average_precision_score(y_te, served_p)),
                           "brier": float(brier_score_loss(y_te, served_p))},
        "operating_threshold": float(best_thr),
        "target_sensitivity": target_sens,
        "confusion_at_threshold": {"tn": int(tn), "fp": int(fp),
                                   "fn": int(fn), "tp": int(tp)},
        "test_sensitivity": float(recall_score(y_te, yhat)),
        "test_precision": float(precision_score(y_te, yhat)),
        "test_specificity": float(tn / (tn + fp)),
        "calibration_curve": calib_json,
    }
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2))
    results.to_csv(REPORTS / "metrics_table.csv", index=False)
    (ARTIFACTS / "operating_threshold.json").write_text(
        json.dumps({"threshold": float(best_thr)}, indent=2))
    print(f"\nSaved reports/evaluation.json and metrics_table.csv")


if __name__ == "__main__":
    main()
