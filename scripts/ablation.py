"""Where does the signal come from? Input-group ablation on the TRAINING participants.

    python ablation.py --cohort framingham|nhanes

Repeated grouped CV (3 x 5 folds) with the gradient-boosting and logistic
models at fixed, moderate settings:
* "without <factor>": that factor's raw inputs are blanked (so every derived
  term is uninformative too). A large drop means the factor carries signal.
* "with <candidate>": an input that was considered and rejected is added back
  (Framingham only), documenting why it is not on the form.
* "published inputs only": only what the published equation uses.
Differences are paired per CV split, so the shared noise cancels.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import cohorts
from config import GROUP, TARGET, dirs
from joblib import Parallel, delayed

from models import make_gbm, make_logreg, single_thread
from train import N_JOBS, _fit_fold, folds

PUBLISHED_INPUTS = {"framingham": ["male", "age", "sysBP", "BPMeds", "currentSmoker", "diabetes", "totChol", "HDL", "BMI"],
                    "nhanes": ["male", "age", "race", "sysBP", "BPMeds", "smoking", "diabetes", "totChol", "HDL"]}


def cv_scores(est_fn, df, c, splits_list, extra_cols=()):
    y = df[TARGET].to_numpy()
    X = c.add_features(df)
    for col in extra_cols:
        X[col] = df[col].astype(float)
    est = est_fn()
    if extra_cols:   # (gbm only) extra columns are unconstrained
        est = est.set_params(monotone_constraints=list(est.get_params()["monotone_constraints"]) + [0] * len(extra_cols))
    est = single_thread(est)
    jobs = [(r, tr, va) for r, splits in enumerate(splits_list) for tr, va in splits]
    out = Parallel(n_jobs=N_JOBS)(delayed(_fit_fold)(est, X, y, tr, va) for _, tr, va in jobs)
    aucs, lls = [], []
    for r in range(len(splits_list)):
        oof = np.zeros(len(y))
        for (rr, _, _), (va, p) in zip(jobs, out):
            if rr == r:
                oof[va] = p
        aucs.append(roc_auc_score(y, oof))
        lls.append(log_loss(y, oof))
    return np.array(aucs), np.array(lls)


def main(name):
    c, out = cohorts.get(name), dirs(name)
    df = pd.read_csv(out["data"] / "train.csv")
    y = df[TARGET].to_numpy()
    X0 = c.add_features(df)
    splits_list = [folds(X0, y, df[GROUP].to_numpy(), seed=s) for s in (11, 12, 13)]
    factors = sorted(set(c.FACTOR_OF.values()) - {"Age", "Sex"})
    learners = {"gbm": lambda: make_gbm(c), "logreg": lambda: make_logreg(c, C=0.1, l1_ratio=0.5)}
    result = {}
    for lname, fn in learners.items():
        print(f"\n== {lname}")
        base_auc, base_ll = cv_scores(fn, df, c, splits_list)
        res = {"all inputs": {"auroc": float(base_auc.mean()), "logloss": float(base_ll.mean())}}
        variants = {f"without {f.lower()}": [k for k, v in c.FACTOR_OF.items() if v == f] for f in factors}
        variants["published inputs only"] = [k for k in c.FEATURES if k not in PUBLISHED_INPUTS[name]]
        for label, blank in variants.items():
            d = df.copy()
            d[blank] = np.nan
            a, l = cv_scores(fn, d, c, splits_list)
            res[label] = {"auroc": float(a.mean()), "d_auroc": float((a - base_auc).mean()),
                          "d_auroc_per_repeat": (a - base_auc).round(4).tolist(), "d_logloss": float((l - base_ll).mean())}
            print(f"  {label:32s} AUROC {a.mean():.4f}  d={1000 * (a - base_auc).mean():+.1f}e-3")
        if lname == "gbm":   # candidate inputs only tested with the tree model (no pipeline changes needed)
            for cand, cols in getattr(c, "CANDIDATES", {}).items():
                a, l = cv_scores(fn, df, c, splits_list, extra_cols=cols)
                res[f"with {cand}"] = {"auroc": float(a.mean()), "d_auroc": float((a - base_auc).mean()),
                                       "d_auroc_per_repeat": (a - base_auc).round(4).tolist(),
                                       "d_logloss": float((l - base_ll).mean())}
                print(f"  with {cand:27s} AUROC {a.mean():.4f}  d={1000 * (a - base_auc).mean():+.1f}e-3")
        result[lname] = res
    (out["rep"] / "ablation.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    main(ap.parse_args().cohort)
