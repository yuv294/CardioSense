"""Compute population reference statistics for the 'compare to cohort' feature.

Produces, from the TRAINING data only, per-feature distributions (percentile
breakpoints) and the per-model risk distribution across the cohort, so the app
can tell a user "your blood pressure is higher than 78% of the cohort" and
"your risk sits in the 85th percentile". Also stores subgroup event rates by
age band and sex for the clinician portal's context panel.

Output: artifacts/population.json
"""
import json
import numpy as np
import pandas as pd
import joblib

from config import DATA_PROCESSED, ARTIFACTS, FEATURES, TARGET

CONTINUOUS_SHOWN = ["age", "sysBP", "diaBP", "totChol", "BMI", "glucose",
                    "heartRate", "cigsPerDay"]


def main():
    X = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    y = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].to_numpy()

    # percentile breakpoints per continuous feature (1..99) so the API can map
    # a value to a percentile without shipping the whole dataset
    pct = {}
    for f in CONTINUOUS_SHOWN:
        vals = X[f].dropna().to_numpy()
        pct[f] = {
            "p": list(range(1, 100)),
            "v": np.percentile(vals, list(range(1, 100))).round(3).tolist(),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
        }

    # cohort risk distribution from the served model (for 'your risk vs others')
    served = joblib.load(ARTIFACTS / "model_served.joblib")
    Xp = np.load(DATA_PROCESSED / "X_train_proc.npy")
    risks = served.predict_proba(Xp)[:, 1]
    risk_dist = {
        "p": list(range(1, 100)),
        "v": np.percentile(risks, list(range(1, 100))).round(4).tolist(),
        "mean": float(risks.mean()),
    }

    # subgroup event rates for the clinician context panel
    df = X.copy()
    df["_y"] = y
    df["age_band"] = pd.cut(df["age"], [0, 40, 50, 60, 70, 200],
                            labels=["<40", "40-49", "50-59", "60-69", "70+"])
    subgroups = {}
    for sex in [0, 1]:
        for band in ["<40", "40-49", "50-59", "60-69", "70+"]:
            sel = df[(df["male"] == sex) & (df["age_band"] == band)]
            if len(sel) >= 10:
                subgroups[f"{'M' if sex else 'F'}_{band}"] = {
                    "n": int(len(sel)),
                    "event_rate": round(float(sel["_y"].mean()), 4),
                }

    out = {
        "n_train": int(len(y)),
        "overall_event_rate": round(float(y.mean()), 4),
        "feature_percentiles": pct,
        "risk_percentiles": risk_dist,
        "subgroups": subgroups,
    }
    (ARTIFACTS / "population.json").write_text(json.dumps(out, indent=2))
    print(f"Saved population.json: {len(pct)} features, {len(subgroups)} subgroups")


if __name__ == "__main__":
    main()
