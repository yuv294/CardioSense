"""Inference service: load each cohort's artifacts once and score patients.

Design rules
  * The model receives only the raw inputs; derived terms come from the SAME
    cohort.add_features used in training, so training and serving cannot drift.
  * A blank optional field stays missing. The models were trained on rows where
    those inputs were genuinely missing, so "unknown" is a supported input and
    nothing is silently invented. Blank required fields are an error.
  * Explanations are exact Shapley values over plain-language factor groups
    (age, blood pressure, cholesterol, ...), measured against the cohort's
    typical person. They add up exactly to the gap between the person's risk
    and the typical person's risk on the log-odds scale, for any model.
  * The published score (Framingham / PCE) is returned alongside, so the user
    and clinician can see both.
"""
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import cohorts  # noqa: E402
from baselines import filled  # noqa: E402
from calibration import apply_platt  # noqa: E402
from models import BASE_MODELS, LABELS  # noqa: E402

DISCLAIMER = ("Educational estimate from research cohorts. Not a medical device; "
              "not for diagnosis or treatment decisions.")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class CohortModel:
    def __init__(self, name):
        self.c = cohorts.get(name)
        art, rep = ROOT / "artifacts" / name, ROOT / "reports" / name
        self.models = joblib.load(art / "models.joblib")
        self.app = json.loads((art / "app.json").read_text())
        self.cv = json.loads((rep / "cv_results.json").read_text())
        self.reports = {k: rep / f"{k}.json" for k in ("evaluation", "nested_cv", "ablation", "cv_results", "data_summary")}
        self.served = self.app["served"]
        self.model = self.models[self.served]
        self.factors = sorted(set(self.c.FACTOR_OF.values()))
        self.ref = self.app["reference"]

    # ---------------------------------------------------------------- inputs
    def clean(self, payload):
        row, missing = {}, []
        for f in self.c.FEATURES:
            v = payload.get(f, None)
            if v is None or v == "":
                if f not in self.c.OPTIONAL:
                    missing.append(self.c.FEATURE_META[f]["label"])
                row[f] = np.nan
                continue
            v = float(v)
            meta = self.c.FEATURE_META[f]
            if meta["kind"] == "number" and not (meta["min"] <= v <= meta["max"]):
                raise ValueError(f"{meta['label']} must be between {meta['min']} and {meta['max']}.")
            row[f] = v
        if missing:
            raise ValueError("Please fill in: " + ", ".join(missing) + ".")
        # consistency: a non-smoker smokes 0 cigarettes
        smoker = row.get("currentSmoker", row.get("smoking") == 2 if "smoking" in row else None)
        if not smoker and "cigsPerDay" in row:
            row["cigsPerDay"] = 0.0
        return row

    def _risk(self, rows, model=None):
        X = self.c.add_features(pd.DataFrame(rows))
        return (model or self.model).predict_proba(X)[:, 1]

    def band(self, p):
        for edge, key, label in self.c.BANDS:
            if p < edge:
                return key, label
        return self.c.BANDS[-1][1], self.c.BANDS[-1][2]

    # ---------------------------------------------------------- explanations
    def shapley(self, row):
        """Exact Shapley values over factor groups (2^G model evaluations in one batch)."""
        G = self.factors
        cols = {g: [f for f, v in self.c.FACTOR_OF.items() if v == g] for g in G}
        masks = [frozenset(s) for k in range(len(G) + 1) for s in combinations(range(len(G)), k)]
        rows = []
        for m in masks:
            r = dict(self.ref)
            for i in m:
                for f in cols[G[i]]:
                    r[f] = row[f]
            rows.append(r)
        v = dict(zip(masks, _logit(self._risk(rows))))
        n, phi = len(G), {}
        w = [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)]
        for i, g in enumerate(G):
            phi[g] = sum(w[len(m)] * (v[m | {i}] - v[m]) for m in masks if i not in m)
        base = float(v[frozenset()])
        items = [{"factor": g, "contribution": round(float(x), 4),
                  "direction": "increases" if x > 0 else "decreases"} for g, x in phi.items()]
        items.sort(key=lambda d: -abs(d["contribution"]))
        return items, 1 / (1 + math.exp(-base))

    def scenarios(self, row, p_now):
        """Risk if each modifiable factor were at a healthy target (model-based, not causal)."""
        has = lambda f: f in row and not np.isnan(row[f])
        changes = []
        smoking_now = row.get("currentSmoker") == 1 or row.get("smoking") == 2
        if smoking_now:
            ch = {"cigsPerDay": 0.0}
            ch.update({"currentSmoker": 0.0} if "currentSmoker" in row else {"smoking": 1.0})
            changes.append(("Quit smoking", ch))
        if has("sysBP") and row["sysBP"] > 120:
            ch = {"sysBP": 120.0}
            if has("diaBP") and row["diaBP"] > 80:
                ch["diaBP"] = 80.0
            changes.append(("Blood pressure to 120/80", ch))
        if has("totChol") and row["totChol"] > 180:
            changes.append(("Total cholesterol to 180", {"totChol": 180.0}))
        if has("BMI") and row["BMI"] > 25:
            ch = {"BMI": 25.0}
            if has("waist"):
                ch["waist"] = round(row["waist"] * 25.0 / row["BMI"], 1)
            changes.append(("Body mass index to 25", ch))
        if has("HbA1c") and row["HbA1c"] > 6.0 and "HbA1c" in self.c.MODIFIABLE:
            changes.append(("HbA1c to 6.0%", {"HbA1c": 6.0}))
        if not changes:
            return []
        combined = {k: v for _, ch in changes for k, v in ch.items()}
        if len(changes) > 1:
            changes.append(("All of the above", combined))
        risks = self._risk([{**row, **ch} for _, ch in changes])
        return [{"label": lab, "changes": ch, "risk": round(float(r), 4), "risk_percent": round(float(r) * 100, 1),
                 "reduction_percent": round(float(p_now - r) * 100, 1)} for (lab, ch), r in zip(changes, risks)]

    # ---------------------------------------------------------------- scoring
    def baseline_risks(self, row):
        X = filled(self.c.add_features(pd.DataFrame([row])))
        out = []
        for name, fn in self.c.BASELINES.items():
            p = float(fn(X)[0])
            if np.isnan(p) or name not in self.cv:
                continue
            out.append({"name": name, "label": LABELS.get(name, name), "risk_percent": round(p * 100, 1),
                        "recalibrated_percent": round(float(apply_platt(p, *self.cv[name]["platt"])) * 100, 1)})
        return out

    def predict(self, payload, mode="patient"):
        row = self.clean(payload)
        p = float(self._risk([row])[0])
        key, label = self.band(p)
        factors, p_ref = self.shapley(row)
        pct = int(np.clip(np.searchsorted(self.app["risk_percentiles"], p) + 1, 1, 99))
        age = row["age"]
        peer_key = f"{'M' if row['male'] == 1 else 'F'}_" + ("<50" if age < 50 else "50-59" if age < 60
                                                            else "60-69" if age < 70 else "70+")
        res = {
            "model": self.c.NAME, "outcome": self.c.OUTCOME_SHORT, "outcome_long": self.c.OUTCOME,
            "risk": round(p, 4), "risk_percent": round(p * 100, 1), "band": key, "band_label": label,
            "band_text": self.c.BAND_TEXT[key],
            "bands": [{"edge": e, "key": k, "label": l} for e, k, l in self.c.BANDS],
            "typical_risk_percent": round(p_ref * 100, 1),
            "factors": factors, "risk_percentile": pct, "peers": self.app["peers"].get(peer_key),
            "scenarios": self.scenarios(row, p),
            "unknown_fields": [self.c.FEATURE_META[f]["label"] for f in self.c.OPTIONAL if np.isnan(row[f])],
            "published_scores": self.baseline_risks(row),
            "disclaimer": DISCLAIMER,
        }
        if mode == "clinician":
            X = self.c.add_features(pd.DataFrame([row]))
            res["model_probabilities"] = sorted(
                [{"model": m, "label": LABELS[m], "risk_percent": round(float(self.models[m].predict_proba(X)[0, 1]) * 100, 1),
                  "served": m == self.served} for m in BASE_MODELS + ["ensemble", "refit"]],
                key=lambda d: -d["risk_percent"])
            res["inputs"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        return res

    def predict_many(self, records):
        out = []
        for i, rec in enumerate(records):
            try:
                row = self.clean(rec)
                p = float(self._risk([row])[0])
                b = self.baseline_risks(row)
                out.append({"row": i + 1, "risk_percent": round(p * 100, 1), "band": self.band(p)[1],
                            **{f"{s['name']}_percent": s["risk_percent"] for s in b}, "error": None})
            except Exception as e:  # report per-row problems instead of failing the whole file
                out.append({"row": i + 1, "error": str(e)})
        return out

    def schema(self):
        return {
            "model": self.c.NAME, "title": self.c.TITLE, "outcome": self.c.OUTCOME, "outcome_short": self.c.OUTCOME_SHORT,
            "groups": [{"key": k, "title": t, "note": n} for k, t, n in self.c.FORM_GROUPS],
            "features": [{"name": f, "optional": f in self.c.OPTIONAL, "modifiable": f in self.c.MODIFIABLE,
                          **{k: v for k, v in self.c.FEATURE_META[f].items()}} for f in self.c.FEATURES],
            "example": self.ref,
        }

    def info(self):
        return {"name": self.c.NAME, "title": self.c.TITLE, "outcome": self.c.OUTCOME,
                "outcome_short": self.c.OUTCOME_SHORT, "source": self.c.SOURCE, "served": self.served,
                "served_label": LABELS[self.served], "n_inputs": len(self.c.FEATURES),
                "n_optional": len(self.c.OPTIONAL), **self.app["summary"]}

    def evidence(self):
        return {k: json.loads(p.read_text()) for k, p in self.reports.items() if p.exists()}


REGISTRY = {}
for _name in ("framingham", "nhanes"):
    if (ROOT / "artifacts" / _name / "app.json").exists():
        REGISTRY[_name] = CohortModel(_name)


def get(name):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name]
