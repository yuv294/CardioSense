"""Evaluation metrics shared by evaluate.py and nested_cv.py.

Follows the TRIPOD+AI reporting checklist (Collins et al., BMJ 2024):
discrimination AND calibration AND clinical usefulness, with uncertainty.

* Bootstrap CIs resample PARTICIPANTS (cluster bootstrap), because a
  Framingham participant contributes up to three correlated rows.
* Paired differences use the same resample for both models, so the shared
  noise cancels; that is what makes small differences detectable.
* Calibration: intercept (calibration-in-the-large) and slope (Van Calster et
  al., BMC Med 2019, "calibration: the Achilles heel of predictive analytics").
* Net benefit from decision curve analysis (Vickers & Elkin, MDM 2006).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from calibration import logit


def calibration_stats(y, p):
    lp = logit(p).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]
    # intercept with the slope fixed at 1 (offset model), one Newton solve
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp[:, 0])))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-9)
    return {"cal_intercept": float(a), "cal_slope": float(slope), "expected_over_observed": float(p.mean() / y.mean())}


def point_metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            **calibration_stats(y, p)}


def cluster_indices(groups, n_boot, seed):
    """Yield row-index arrays of participant-level bootstrap resamples."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([rows_of[i] for i in pick])


def bootstrap(y, preds: dict, groups, pairs, n_boot=1000, seed=0):
    """CIs for AUROC/Brier of every model and for paired differences (a - b)."""
    stats = {m: {"auroc": [], "brier": []} for m in preds}
    diffs = {f"{a}-vs-{b}": {"auroc": [], "brier": [], "logloss": []} for a, b in pairs}
    for idx in cluster_indices(groups, n_boot, seed):
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        cur = {}
        for m, p in preds.items():
            pp = p[idx]
            cur[m] = (roc_auc_score(yy, pp), brier_score_loss(yy, pp), log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6)))
            stats[m]["auroc"].append(cur[m][0])
            stats[m]["brier"].append(cur[m][1])
        for a, b in pairs:
            k = f"{a}-vs-{b}"
            for j, key in enumerate(("auroc", "brier", "logloss")):
                diffs[k][key].append(cur[a][j] - cur[b][j])

    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out_m = {m: {f"{k}_ci": ci(v) for k, v in s.items()} for m, s in stats.items()}
    out_d = {}
    for k, s in diffs.items():
        a = np.array(s["auroc"])
        out_d[k] = {"d_auroc": float(a.mean()), "d_auroc_ci": ci(a),
                    # two-sided bootstrap p-value for "no difference in AUROC"
                    "p_auroc": float(min(1.0, 2 * min((a <= 0).mean(), (a >= 0).mean()))),
                    "d_brier": float(np.mean(s["brier"])), "d_brier_ci": ci(s["brier"]),
                    "d_logloss": float(np.mean(s["logloss"])), "d_logloss_ci": ci(s["logloss"])}
    return out_m, out_d


def net_benefit(y, p, thresholds):
    n, out = len(y), []
    for t in thresholds:
        flag = p >= t
        tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
        out.append(float(tp / n - fp / n * t / (1 - t)))
    return out


def calibration_curve(y, p, bins=10):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    pts = [{"predicted": float(p[which == b].mean()), "observed": float(y[which == b].mean()), "n": int((which == b).sum())}
           for b in range(len(edges) - 1) if (which == b).any()]
    return pts


def at_threshold(y, p, t):
    flag = p >= t
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged_share": float(flag.mean()),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "ppv": tp / max(tp + fp, 1)}
