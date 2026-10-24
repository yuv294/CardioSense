"""Step 4 (secondary analysis): nested cross-validation over the whole cohort.

    python nested_cv.py --cohort framingham|nhanes

The locked test set holds only 20% of the events (77 CVD deaths in NHANES), so
its confidence intervals are wide. Here the ENTIRE training procedure of
train.fit_all (tuning, one-SE rule, calibration, ensemble, served-model choice)
is repeated inside each of 5 outer folds, and each participant is predicted by
the run that never saw them. Every event then counts towards the comparison.

It is run after evaluate.py, and nothing in the pipeline was changed in
response to it (otherwise it would stop being an honest estimate).
"""
import argparse
import json

import numpy as np
import pandas as pd

import cohorts
from baselines import filled
from calibration import apply_platt
from config import GROUP, TARGET, dirs
from metrics import bootstrap, point_metrics
from models import BASE_MODELS
from train import fit_all, folds


def main(name, n_boot):
    c, out = cohorts.get(name), dirs(name)
    df = pd.concat([pd.read_csv(out["data"] / f) for f in ("train.csv", "test.csv")], ignore_index=True)
    y, groups = df[TARGET].to_numpy(), df[GROUP].to_numpy()
    X = c.add_features(df)
    keys = BASE_MODELS + ["ensemble", "refit", "served"]
    pred = {k: np.zeros(len(df)) for k in keys}
    base_names = [b for b in c.BASELINES if not np.isnan(c.BASELINES[b](filled(X))).any()]
    for b in base_names:
        pred[f"{b}_recal"] = np.zeros(len(df))
    served_per_fold = []
    for k, (tr, te) in enumerate(folds(X, y, groups, seed=7)):
        print(f"outer fold {k + 1}/5", flush=True)
        fitted, res, _ = fit_all(c, df.iloc[tr].reset_index(drop=True), verbose=False)
        Xte = X.iloc[te]
        for m in BASE_MODELS + ["ensemble", "refit"]:
            pred[m][te] = fitted[m].predict_proba(Xte)[:, 1]
        pred["served"][te] = pred[res["_served"]][te]
        served_per_fold.append(res["_served"])
        for b in base_names:
            pred[f"{b}_recal"][te] = apply_platt(c.BASELINES[b](filled(Xte)), *res[b]["platt"])
    for b in base_names:
        pred[b] = c.BASELINES[b](filled(X))

    base = c.PRIMARY_BASELINE
    pairs = [("served", base), ("served", f"{base}_recal"), ("served", "refit"), ("refit", base)] + \
            [(m, base) for m in BASE_MODELS + ["ensemble"]]
    boot_m, boot_d = bootstrap(y, pred, groups, pairs, n_boot=n_boot, seed=1)
    rows = {m: {**point_metrics(y, p), **boot_m[m]} for m, p in pred.items()}
    report = {"cohort": name, "rows": int(len(y)), "events": int(y.sum()), "served_per_fold": served_per_fold,
              "metrics": rows, "paired": boot_d, "n_bootstrap": n_boot}
    (out["rep"] / "nested_cv.json").write_text(json.dumps(report, indent=2))
    print(f"\n=== nested CV, {name}: {len(y)} rows, {int(y.sum())} events; served per fold: {served_per_fold}")
    for m, r in sorted(rows.items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"{m:16s} AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]  Brier {r['brier']:.4f}"
              f"  E/O {r['expected_over_observed']:.2f}")
    for k, d in boot_d.items():
        print(f"{k:24s} dAUROC={d['d_auroc']:+.3f} [{d['d_auroc_ci'][0]:+.3f},{d['d_auroc_ci'][1]:+.3f}] p={d['p_auroc']:.3f}"
              f"  dBrier={d['d_brier']:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    main(a.cohort, a.boot)
