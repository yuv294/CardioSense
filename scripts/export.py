"""Step 5: write what the API needs to serve a cohort's model.

    python export.py --cohort framingham|nhanes

artifacts/<cohort>/app.json holds
* reference: the "typical person" (training medians / modes) that explanations
  are measured against;
* risk_percentiles: out-of-fold risks of the training participants, so a new
  risk can be placed in the cohort's distribution;
* peers: average predicted risk and observed event rate by sex and age band;
* summary: headline numbers from the locked test set for the app.
"""
import argparse
import json

import numpy as np
import pandas as pd

import cohorts
from config import TARGET, dirs


def age_band(a):
    return "<50" if a < 50 else "50-59" if a < 60 else "60-69" if a < 70 else "70+"


def main(name):
    c, out = cohorts.get(name), dirs(name)
    train = pd.read_csv(out["data"] / "train.csv")
    oof = pd.read_csv(out["art"] / "oof_predictions.csv")
    cv = json.loads((out["rep"] / "cv_results.json").read_text())
    ev = json.loads((out["rep"] / "evaluation.json").read_text())
    served = cv["_served"]
    risk = oof[served].to_numpy()

    ref = {}
    for f in c.FEATURES:
        meta = c.FEATURE_META[f]
        v = train[f].dropna()
        ref[f] = float(v.mode().iloc[0]) if meta["kind"] in ("binary", "select") else float(round(v.median(), 1))
    if "cigsPerDay" in ref:      # the typical person in both cohorts does not smoke
        ref["cigsPerDay"] = 0.0

    peers = {}
    t = train.assign(risk=risk, band=train["age"].map(age_band), sex=np.where(train["male"] == 1, "M", "F"))
    for (sex, band), g in t.groupby(["sex", "band"]):
        peers[f"{sex}_{band}"] = {"n": int(len(g)), "mean_risk": float(g["risk"].mean()),
                                  "observed_rate": float(g[TARGET].mean())}

    m, b = ev["metrics"], ev["primary_baseline"]
    pair = ev["paired"][f"{served}-vs-{b}"]
    summary = {"served": served, "baseline": b, "auroc": m[served]["auroc"], "auroc_ci": m[served]["auroc_ci"],
               "baseline_auroc": m[b]["auroc"], "baseline_auroc_ci": m[b]["auroc_ci"],
               "d_auroc": pair["d_auroc"], "d_auroc_ci": pair["d_auroc_ci"], "p_auroc": pair["p_auroc"],
               "n_test": ev["n_test"], "events_test": ev["events_test"]}
    nested_path = out["rep"] / "nested_cv.json"
    if nested_path.exists():   # the more precise estimate: every event, whole tuning procedure inside each fold
        nest = json.loads(nested_path.read_text())
        d = nest["paired"][f"served-vs-{b}"]
        summary["nested"] = {"d_auroc": d["d_auroc"], "d_auroc_ci": d["d_auroc_ci"], "p_auroc": d["p_auroc"],
                             "d_brier": d["d_brier"], "d_brier_ci": d["d_brier_ci"], "events": nest["events"],
                             "auroc": nest["metrics"]["served"]["auroc"], "baseline_auroc": nest["metrics"][b]["auroc"]}
    app = {"name": name, "title": c.TITLE, "outcome": c.OUTCOME, "outcome_short": c.OUTCOME_SHORT, "source": c.SOURCE,
           "served": served, "reference": ref, "risk_percentiles": np.percentile(risk, np.arange(1, 100)).tolist(),
           "peers": peers, "event_rate": float(train[TARGET].mean()), "summary": summary}
    (out["art"] / "app.json").write_text(json.dumps(app, indent=2))
    print(f"wrote {out['art'] / 'app.json'} (served: {served})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    main(ap.parse_args().cohort)
