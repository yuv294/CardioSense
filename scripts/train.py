"""Step 2: tune, calibrate and fit every model on the TRAINING participants only.

    python train.py --cohort framingham|nhanes

* Grouped, stratified 5-fold CV (all rows of a person stay in one fold).
* Hyperparameters picked on CV log-loss (a proper scoring rule: it rewards
  calibrated probabilities, not just ranking). The one-standard-error rule then
  takes the most regularised setting statistically tied with the best, which
  guards against over-tuning to CV noise (Hastie et al., ESL 7.10).
* Each tuned model is refitted on all training rows and recalibrated (Platt)
  on its own out-of-fold predictions.
* The ensemble averages the four calibrated models (no fitted weights).
* The served model is chosen on CV log-loss only; the test set is not opened.

fit_all() is also called inside every outer fold by nested_cv.py, so the
nested estimate measures this exact procedure, tuning included.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import cohorts
from baselines import filled
from calibration import Calibrated, MeanEnsemble, apply_platt, fit_platt
from config import CV_FOLDS, GROUP, RANDOM_STATE, TARGET, dirs
from models import BASE_MODELS, SPECS, sample_space, single_thread

N_SETTINGS = {"logreg": 10, "rf": 12, "gbm": 16, "mlp": 12, "refit": 1}
# higher = more regularised; used by the one-standard-error rule
SIMPLICITY = {
    "logreg": lambda s: -s["C"],
    "rf": lambda s: s["min_samples_leaf"],
    "gbm": lambda s: -s["num_leaves"] * s["n_estimators"] * s["learning_rate"] + s["reg_lambda"],
    "mlp": lambda s: s["alpha"] - sum(s["hidden"]) / 100,
    "refit": lambda s: 0,
}


def folds(X, y, groups, seed=RANDOM_STATE):
    return list(StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=seed).split(X, y, groups))


N_JOBS = int(os.environ.get("CS_JOBS", max(1, (os.cpu_count() or 2) - 2)))


def _fit_fold(est, X, y, tr, va):
    return va, clone(est).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]


def oof_predict(est, X, y, splits):
    oof, fold_losses = np.zeros(len(y)), []
    for tr, va in splits:
        m = clone(est).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        fold_losses.append(log_loss(y[va], oof[va], labels=[0, 1]))
    return oof, np.array(fold_losses)


def tune(c, name, X, y, splits, rng, verbose=True):
    """Every (setting, fold) fit runs in parallel; models are single-threaded inside."""
    spec, trials = SPECS[name], []
    settings = sample_space(spec["space"], N_SETTINGS[name], rng)
    t0 = time.time()
    jobs = [(i, spec["make"](c, **s), tr, va) for i, s in enumerate(settings) for tr, va in splits]
    out = Parallel(n_jobs=N_JOBS)(delayed(_fit_fold)(single_thread(est), X, y, tr, va) for _, est, tr, va in jobs)
    for i, s in enumerate(settings):
        oof, fl = np.zeros(len(y)), []
        for (j, _, _, _), (va, p) in zip(jobs, out):
            if j == i:
                oof[va] = p
                fl.append(log_loss(y[va], p, labels=[0, 1]))
        fl = np.array(fl)
        trials.append({"setting": s, "logloss": fl.mean(), "se": fl.std(ddof=1) / np.sqrt(len(fl)),
                       "auroc": roc_auc_score(y, oof), "oof": oof})
        if verbose:
            print(f"  {name:7s} {s}  logloss={fl.mean():.4f}  auroc={trials[-1]['auroc']:.4f}", flush=True)
    if verbose:
        print(f"  ({len(jobs)} fits in {time.time() - t0:.0f}s)", flush=True)
    best = min(trials, key=lambda t: t["logloss"])
    ok = [t for t in trials if t["logloss"] <= best["logloss"] + best["se"]]
    return max(ok, key=lambda t: SIMPLICITY[name](t["setting"])), trials


def fit_all(c, df, verbose=True):
    """Tune + calibrate + refit every model on df. Returns (fitted, results, oof_calibrated)."""
    X, y = c.add_features(df), df[TARGET].to_numpy()
    splits = folds(X, y, df[GROUP].to_numpy())
    rng = np.random.default_rng(RANDOM_STATE)
    fitted, results, oof_cal = {}, {}, {}
    for name in BASE_MODELS + ["refit"]:
        if verbose:
            print(f"\n== {name}")
        chosen, trials = tune(c, name, X, y, splits, rng, verbose)
        a, b = fit_platt(chosen["oof"], y)
        cal = np.zeros(len(y))               # honest CV estimate: Platt map refitted inside each fold
        for tr, va in splits:
            aa, bb = fit_platt(chosen["oof"][tr], y[tr])
            cal[va] = apply_platt(chosen["oof"][va], aa, bb)
        oof_cal[name] = cal
        fitted[name] = Calibrated(SPECS[name]["make"](c, **chosen["setting"]).fit(X, y), a, b)
        results[name] = {"setting": {k: list(v) if isinstance(v, tuple) else v for k, v in chosen["setting"].items()},
                         "cv_logloss": float(log_loss(y, cal)), "cv_auroc": float(roc_auc_score(y, cal)),
                         "train_auroc": float(roc_auc_score(y, fitted[name].predict_proba(X)[:, 1])),
                         "platt": [a, b], "n_settings_tried": len(trials)}
        if verbose:
            r = results[name]
            print(f"  -> {chosen['setting']}  CV auroc={r['cv_auroc']:.4f}  train auroc={r['train_auroc']:.4f}", flush=True)

    oof_cal["ensemble"] = np.mean([oof_cal[m] for m in BASE_MODELS], axis=0)
    fitted["ensemble"] = MeanEnsemble({m: fitted[m] for m in BASE_MODELS})
    results["ensemble"] = {"cv_logloss": float(log_loss(y, oof_cal["ensemble"])),
                           "cv_auroc": float(roc_auc_score(y, oof_cal["ensemble"])),
                           "train_auroc": float(roc_auc_score(y, fitted["ensemble"].predict_proba(X)[:, 1]))}

    # published equations on the same rows (fixed formulas) and their recalibrated versions
    Xf = filled(X)
    for bname, fn in c.BASELINES.items():
        p = fn(Xf)
        ok = ~np.isnan(p)
        a, b = fit_platt(p[ok], y[ok])
        results[bname] = {"cv_auroc": float(roc_auc_score(y[ok], p[ok])), "cv_logloss": float(log_loss(y[ok], p[ok])),
                          "platt": [a, b], "rows": int(ok.sum())}
        oof_cal[bname] = p
    served = min(BASE_MODELS + ["ensemble"], key=lambda m: results[m]["cv_logloss"])
    results["_served"] = served
    return fitted, results, oof_cal


def main(name):
    c, out = cohorts.get(name), dirs(name)
    df = pd.read_csv(out["data"] / "train.csv")
    fitted, results, oof = fit_all(c, df)
    print(f"\nServed model (lowest CV log-loss): {results['_served']}")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k:10s} CV auroc={v['cv_auroc']:.4f}  CV logloss={v['cv_logloss']:.4f}")
    joblib.dump(fitted, out["art"] / "models.joblib")
    pd.DataFrame({GROUP: df[GROUP], "y": df[TARGET], **oof}).to_csv(out["art"] / "oof_predictions.csv", index=False)
    (out["rep"] / "cv_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    main(ap.parse_args().cohort)
