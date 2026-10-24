"""Step 3: open the locked test set ONCE and judge everything on it.

    python evaluate.py --cohort framingham|nhanes

Compares the tuned models with the published equation(s), raw and
recalibrated on the training data, and with the published formula refitted.
Writes reports/<cohort>/evaluation.json, used by the app's Evidence page.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import cohorts
from baselines import filled
from calibration import apply_platt
from config import GROUP, TARGET, dirs
from metrics import at_threshold, bootstrap, calibration_curve, net_benefit, point_metrics
from models import BASE_MODELS, LABELS


def age_band(a):
    return pd.cut(a, [0, 50, 60, 70, 200], right=False, labels=["<50", "50-59", "60-69", "70+"]).astype(str)


def predictions(c, fitted, cv, df):
    X = c.add_features(df)
    preds = {m: fitted[m].predict_proba(X)[:, 1] for m in BASE_MODELS + ["ensemble", "refit"]}
    Xf = filled(X)
    for b, fn in c.BASELINES.items():
        p = fn(Xf)
        if np.isnan(p).any():          # e.g. the lab-only Framingham score; reported on its own subset
            continue
        preds[b] = p
        preds[f"{b}_recal"] = apply_platt(p, *cv[b]["platt"])
    return preds


def main(name, n_boot):
    c, out = cohorts.get(name), dirs(name)
    test = pd.read_csv(out["data"] / "test.csv")
    fitted = joblib.load(out["art"] / "models.joblib")
    cv = json.loads((out["rep"] / "cv_results.json").read_text())
    served, base = cv["_served"], c.PRIMARY_BASELINE
    y, groups = test[TARGET].to_numpy(), test[GROUP].to_numpy()
    preds = predictions(c, fitted, cv, test)

    rows = {m: point_metrics(y, p) for m, p in preds.items()}
    pairs = [(served, base), (served, f"{base}_recal"), (served, "refit"), ("refit", base)]
    pairs += [(m, base) for m in BASE_MODELS + ["ensemble"] if m != served]
    boot_m, boot_d = bootstrap(y, preds, groups, pairs, n_boot=n_boot)
    for m in rows:
        rows[m].update(boot_m[m])
        rows[m]["label"] = LABELS.get(m, LABELS.get(m.replace("_recal", ""), m) + (" (recalibrated)" if m.endswith("_recal") else ""))

    lo, hi = c.DCA_RANGE
    ts = np.round(np.linspace(lo, hi, 40), 4)
    dca = {"thresholds": ts.tolist(), "treat_all": net_benefit(y, np.ones_like(y, dtype=float), ts),
           **{m: net_benefit(y, preds[m], ts) for m in (served, base, f"{base}_recal", "refit")}}
    high = c.BANDS[-2][0]               # lower edge of the "high" band
    decision = {m: at_threshold(y, preds[m], high) for m in (served, base, f"{base}_recal")}

    sub = pd.DataFrame({"sex": np.where(test["male"] == 1, "men", "women"), "age": age_band(test["age"])})
    for s in c.STRATA:
        sub[s] = test[s].astype(str)
    if "race" in test:
        sub["race"] = test["race"].map({3: "White", 4: "Black", 1: "Hispanic", 2: "Hispanic", 5: "Other"})
    subgroups = []
    for col in sub.columns:
        for level, idx in sub.groupby(col).groups.items():
            yy = y[idx]
            if yy.sum() < 10 or yy.sum() == len(yy):
                continue
            subgroups.append({"by": col, "level": level, "n": int(len(idx)), "events": int(yy.sum()),
                              **{f"auroc_{m}": point_metrics(yy, preds[m][idx])["auroc"] for m in (served, base)},
                              **{f"eo_{m}": float(preds[m][idx].mean() / yy.mean()) for m in (served, base, f"{base}_recal")}})

    extra = {}
    if "frs_lab" in c.BASELINES:        # lab Framingham where HDL is known (exam 3)
        X = c.add_features(test)
        p_lab = c.BASELINES["frs_lab"](filled(X))
        ok = ~np.isnan(p_lab)
        extra["hdl_known_subset"] = {"n": int(ok.sum()), "events": int(y[ok].sum()),
                                     "auroc_frs_lab": point_metrics(y[ok], p_lab[ok])["auroc"],
                                     f"auroc_{served}": point_metrics(y[ok], preds[served][ok])["auroc"]}

    report = {
        "cohort": name, "title": c.TITLE, "outcome": c.OUTCOME, "source": c.SOURCE,
        "n_test": int(len(y)), "participants_test": int(len(np.unique(groups))), "events_test": int(y.sum()),
        "event_rate_test": float(y.mean()), "served_model": served, "primary_baseline": base,
        "metrics": rows, "paired": boot_d, "decision_curve": dca, "at_high_risk_threshold": decision,
        "calibration_curves": {m: calibration_curve(y, preds[m]) for m in (served, base, f"{base}_recal")},
        "subgroups": subgroups, **extra, "n_bootstrap": n_boot,
        "overfitting_check": {m: {"train_auroc": cv[m].get("train_auroc"), "cv_auroc": cv[m]["cv_auroc"],
                                  "test_auroc": rows[m]["auroc"]} for m in BASE_MODELS + ["ensemble", "refit"]},
    }
    (out["rep"] / "evaluation.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== {c.TITLE}: locked test set, {len(y)} rows, {int(y.sum())} events ===")
    print(f"{'model':16s} {'AUROC':>7s} {'95% CI':>15s} {'Brier':>7s} {'E/O':>5s} {'slope':>6s}")
    for m, r in sorted(rows.items(), key=lambda kv: -kv[1]["auroc"]):
        print(f"{m:16s} {r['auroc']:7.3f}  [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}] {r['brier']:7.4f} "
              f"{r['expected_over_observed']:5.2f} {r['cal_slope']:6.2f}")
    print()
    for k, d in boot_d.items():
        print(f"{k:28s} dAUROC={d['d_auroc']:+.3f} [{d['d_auroc_ci'][0]:+.3f},{d['d_auroc_ci'][1]:+.3f}] p={d['p_auroc']:.3f}"
              f"   dBrier={d['d_brier']:+.4f} [{d['d_brier_ci'][0]:+.4f},{d['d_brier_ci'][1]:+.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    main(a.cohort, a.boot)
