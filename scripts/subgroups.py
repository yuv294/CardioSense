"""Point 4: is the model trustworthy for everyone, and is it useful?  (not just on average)

Uses the out-of-fold predictions saved by ladder.py (development rows only, every prediction made
by a model that never saw that patient). Two questions per group (sex, age band):
  * calibration: when the model says X%, do about X% develop disease?
  * discrimination: AUROC inside the group
and one overall question, decision-curve net benefit: across screening thresholds, does acting on
the model's flag do more good than treating everyone, treating no-one, or using Framingham?

Calibration of the models is cross-fitted (each patient's calibrated value comes from a calibrator
fitted on the other folds) so the numbers are not flattered. Framingham is shown as published
("raw") and after a 2-parameter recalibration, which is the fairest version of the clinical score.
Writes reports/subgroups.json and reports/subgroups.png.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from config import DATA_RAW, REPORTS


def crossfit(p, y, kind):
    out = np.zeros_like(p)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=3).split(p, y):
        if kind == "isotonic":
            out[te] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr]).predict(p[te])
        else:
            z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
            out[te] = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr], y[tr]).predict_proba(z[te])[:, 1]
    return out


def slope(p, y):
    z = logit(np.clip(p, 1e-4, 1 - 1e-4)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(z, y).coef_[0, 0])


def net_benefit(p, y, t):
    flag = p >= t
    tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
    return tp / len(y) - fp / len(y) * t / (1 - t)


def main():
    z = np.load(REPORTS / "ladder_oof.npz")
    y, idx = z["y"], z["idx"]
    raw = pd.read_csv(DATA_RAW).iloc[idx].reset_index(drop=True)
    avg = lambda k: z[k].mean(0)
    models = {
        "CardioSense (S3, calibrated)": crossfit(avg("S3"), y, "isotonic"),
        "Original 23-feature model (S2, calibrated)": crossfit(avg("S2"), y, "isotonic"),
        "Framingham (as published)": avg("Framingham"),
        "Framingham (recalibrated)": crossfit(avg("Framingham"), y, "logistic"),
    }
    groups = {"All": np.ones(len(y), bool), "Women": raw.male.values == 0, "Men": raw.male.values == 1,
              "Age under 50": raw.age.values < 50, "Age 50 to 59": (raw.age.values >= 50) & (raw.age.values < 60),
              "Age 60 and over": raw.age.values >= 60}
    out = {"n": int(len(y)), "events": int(y.sum()), "groups": {}, "decision_curve": {}}
    for g, m in groups.items():
        yy = y[m]
        row = {"n": int(m.sum()), "events": int(yy.sum()), "observed_rate": float(yy.mean()), "models": {}}
        for k, p in models.items():
            pp = p[m]
            row["models"][k] = {"mean_predicted": float(pp.mean()), "auroc": float(roc_auc_score(yy, pp)),
                                "brier": float(brier_score_loss(yy, pp)), "calibration_slope": slope(pp, yy)}
        out["groups"][g] = row
    ts = np.round(np.arange(0.05, 0.401, 0.01), 2)
    out["decision_curve"]["thresholds"] = [float(t) for t in ts]
    for k, p in models.items():
        out["decision_curve"][k] = [float(net_benefit(p, y, t)) for t in ts]
    out["decision_curve"]["Treat everyone"] = [float(y.mean() - (1 - y.mean()) * t / (1 - t)) for t in ts]
    (REPORTS / "subgroups.json").write_text(json.dumps(out, indent=2))

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    names = list(groups)
    xs = np.arange(len(names))
    obs = [out["groups"][g]["observed_rate"] * 100 for g in names]
    ax[0].bar(xs - 0.27, obs, 0.27, color="#2B3440", label="Observed")
    ax[0].bar(xs, [out["groups"][g]["models"]["CardioSense (S3, calibrated)"]["mean_predicted"] * 100 for g in names],
              0.27, color="#0E7C86", label="CardioSense")
    ax[0].bar(xs + 0.27, [out["groups"][g]["models"]["Framingham (as published)"]["mean_predicted"] * 100 for g in names],
              0.27, color="#E0A030", label="Framingham (published)")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"{g}\n(n={out['groups'][g]['n']})" for g in names], fontsize=8)
    ax[0].set_ylabel("Average 10-year risk (%)")
    ax[0].set_title("Predicted vs observed risk by group")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].spines[["top", "right"]].set_visible(False)
    dc = out["decision_curve"]
    ax[1].plot(ts * 100, dc["CardioSense (S3, calibrated)"], color="#0E7C86", lw=2.2, label="CardioSense")
    ax[1].plot(ts * 100, dc["Framingham (recalibrated)"], color="#E0A030", lw=2, label="Framingham (recalibrated)")
    ax[1].plot(ts * 100, dc["Treat everyone"], color="#8896A6", lw=1.5, ls="--", label="Treat everyone")
    ax[1].axhline(0, color="black", lw=1, label="Treat no one")
    ax[1].set_ylim(-0.01, 0.08)
    ax[1].set_xlabel("Screening threshold (%)")
    ax[1].set_ylabel("Net benefit")
    ax[1].set_title("Decision curve")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "subgroups.png", dpi=170)
    for g in names:
        r = out["groups"][g]
        print(f"{g:16s} n={r['n']:5d} ev={r['events']:4d} observed {r['observed_rate']:.3f} | " + " | ".join(
            f"{k.split(' (')[0][:12]} pred {v['mean_predicted']:.3f} auc {v['auroc']:.3f} slope {v['calibration_slope']:.2f}"
            for k, v in r["models"].items()))


if __name__ == "__main__":
    main()
