"""Inference service: load artifacts once, score one patient honestly.

Design rules (these are also viva talking points):
  * The model is fed ONLY the 15 real inputs, then the SAME engineered features
    used in training are derived from them (shared features.py). Nothing is
    fabricated from an unrelated field.
  * A blank optional field is filled with the TRAINING median/mode and reported
    in `assumed_fields`, so the user always sees what was guessed.
  * Preprocessing is the SAME fitted ColumnTransformer from training.
  * The served model is calibrated, so the probability is trustworthy.
  * Two detail levels: patient (risk + top factors + population) and clinician
    (all models, full contributions, subgroup context, explicit threshold).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ART = Path(__file__).resolve().parents[2].parent / "artifacts"
sys.path.insert(0, str(ART.parent / "scripts"))

from config import FEATURES, FEATURE_META, OPTIONAL_FEATURES  # noqa: E402
from features import add_engineered_features, add_structure_features  # noqa: E402

_PRE = joblib.load(ART / "preprocessor.joblib")
_SERVED = joblib.load(ART / "model_served.joblib")
_DEFAULTS = json.loads((ART / "feature_defaults.json").read_text())
_EXPLAINER = json.loads((ART / "explainer.json").read_text())
_THR = json.loads((ART / "operating_threshold.json").read_text())["threshold"]
_POP = json.loads((ART / "population.json").read_text())

# all base models for the clinician multi-model view
_ALL_MODELS = {}
for _name in ["logreg", "rf", "hgb", "mlp", "stack"]:
    _p = ART / f"model_{_name}.joblib"
    if _p.exists():
        _ALL_MODELS[_name] = joblib.load(_p)

_PRETTY = {
    "age": "Age", "sysBP": "Systolic BP", "diaBP": "Diastolic BP",
    "male": "Male sex", "cigsPerDay": "Cigarettes/day", "currentSmoker": "Current smoker",
    "glucose": "Blood glucose", "totChol": "Total cholesterol", "BMI": "Body mass index",
    "prevalentHyp": "Hypertension", "BPMeds": "BP medication", "diabetes": "Diabetes",
    "prevalentStroke": "Prior stroke", "heartRate": "Heart rate", "education": "Education",
    "pulse_pressure": "Pulse pressure", "map": "Mean arterial pressure",
    "bp_age": "BP x age", "smoker_pack": "Smoking load", "chol_age": "Cholesterol x age",
    "hyp_and_diab": "Hypertension + diabetes", "high_glucose": "High glucose",
    "age_sq": "Age (non-linear)",
    "ln_age": "Age (log scale)", "ln_sbp": "Systolic BP (log scale)", "ln_bmi": "BMI (log scale)",
    "male_x_ln_age": "Age in men", "male_x_ln_sbp": "Systolic BP in men",
    "male_x_smoker": "Smoking in men", "male_x_diabetes": "Diabetes in men",
    "treated_x_ln_sbp": "BP on treatment",
}
_MODEL_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest",
                 "hgb": "Gradient Boosting", "mlp": "Neural Network",
                 "stack": "Stacking Ensemble"}


def _band(p):
    if p < 0.10:
        return "low"
    if p < 0.20:
        return "borderline"
    return "high"


def _complete_raw(payload):
    """Fill the 15 raw inputs, tracking which were assumed."""
    assumed, row = [], {}
    for f in FEATURES:
        v = payload.get(f, None)
        if v is None or v == "":
            row[f] = _DEFAULTS[f]
            assumed.append(f)
        else:
            row[f] = float(v)
    return row, assumed


def _design_row(raw_row):
    """Raw dict -> engineered, preprocessed model input (1 x n)."""
    df = add_structure_features(add_engineered_features(pd.DataFrame([raw_row])))
    # transform expects the raw+engineered columns the ColumnTransformer saw
    return _PRE.transform(df)


def _contributions(raw_row):
    """Exact signed contribution per feature from the linear served model."""
    df = add_structure_features(add_engineered_features(pd.DataFrame([raw_row])))
    coef, mean, scale = _EXPLAINER["coef"], _EXPLAINER["mean"], _EXPLAINER["scale"]
    out = []
    for f in _EXPLAINER["processed_columns"]:
        x = float(df[f].iloc[0])
        s = scale[f] if scale[f] else 1.0
        contrib = coef[f] * ((x - mean[f]) / s)
        out.append({"feature": f, "label": _PRETTY.get(f, f),
                    "contribution": round(contrib, 4),
                    "direction": "increases" if contrib > 0 else "decreases"})
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out


# Patients should see "Age" once, not five overlapping age terms. Contributions are additive,
# so summing each family of features into one clinical factor is exact, not an approximation.
_FACTOR_OF = {
    "age": "Age", "age_sq": "Age", "ln_age": "Age", "male_x_ln_age": "Age",
    "sysBP": "Blood pressure", "diaBP": "Blood pressure", "pulse_pressure": "Blood pressure",
    "map": "Blood pressure", "ln_sbp": "Blood pressure", "male_x_ln_sbp": "Blood pressure",
    "treated_x_ln_sbp": "Blood pressure", "bp_age": "Blood pressure", "prevalentHyp": "Blood pressure",
    "BPMeds": "Blood pressure",
    "currentSmoker": "Smoking", "cigsPerDay": "Smoking", "smoker_pack": "Smoking", "male_x_smoker": "Smoking",
    "totChol": "Cholesterol", "chol_age": "Cholesterol",
    "glucose": "Blood glucose and diabetes", "high_glucose": "Blood glucose and diabetes",
    "diabetes": "Blood glucose and diabetes", "male_x_diabetes": "Blood glucose and diabetes",
    "hyp_and_diab": "Blood glucose and diabetes",
    "BMI": "Body mass index", "ln_bmi": "Body mass index", "male": "Sex",
}


def _by_factor(contribs):
    out = {}
    for c in contribs:
        name = _FACTOR_OF.get(c["feature"], c["label"])
        out[name] = out.get(name, 0.0) + c["contribution"]
    items = [{"feature": n, "label": n, "contribution": round(v, 4),
              "direction": "increases" if v > 0 else "decreases"} for n, v in out.items()]
    return sorted(items, key=lambda d: abs(d["contribution"]), reverse=True)


def _percentile(feature, value):
    """Map a raw value to its cohort percentile."""
    pp = _POP["feature_percentiles"].get(feature)
    if not pp:
        return None
    v = np.array(pp["v"])
    idx = int(np.searchsorted(v, value))
    return max(1, min(99, idx + 1))


def _risk_percentile(prob):
    v = np.array(_POP["risk_percentiles"]["v"])
    idx = int(np.searchsorted(v, prob))
    return max(1, min(99, idx + 1))


def _subgroup(raw_row):
    sex = "M" if raw_row["male"] else "F"
    age = raw_row["age"]
    band = ("<40" if age < 40 else "40-49" if age < 50 else "50-59" if age < 60
            else "60-69" if age < 70 else "70+")
    key = f"{sex}_{band}"
    sg = _POP["subgroups"].get(key)
    return {"key": key, **sg} if sg else None


def predict(payload: dict, mode: str = "patient") -> dict:
    raw, assumed = _complete_raw(payload)
    Xp = _design_row(raw)
    prob = float(_SERVED.predict_proba(Xp)[0, 1])
    contribs = _contributions(raw)

    result = {
        "risk": round(prob, 4),
        "risk_percent": round(prob * 100, 1),
        "band": _band(prob),
        "flagged_high_risk": bool(prob >= _THR),
        "threshold": round(_THR, 4),
        "threshold_percent": round(_THR * 100, 1),
        "assumed_fields": assumed,
        "top_factors": [
            {"feature": c["feature"], "label": c["label"],
             "direction": c["direction"], "weight": abs(c["contribution"])}
            for c in _by_factor(contribs)[:5]
        ],
        "risk_percentile": _risk_percentile(prob),
        "disclaimer": ("Educational estimate from the Framingham teaching cohort. "
                       "Not a medical device; not for diagnosis or treatment."),
    }

    # population comparison for the patient portal
    result["population"] = {
        "cohort_event_rate": _POP["overall_event_rate"],
        "your_features": [
            {"feature": f, "label": _PRETTY.get(f, f),
             "value": raw[f], "percentile": _percentile(f, raw[f])}
            for f in ["age", "sysBP", "totChol", "BMI", "glucose"]
            if _percentile(f, raw[f]) is not None
        ],
    }

    if mode == "clinician":
        # every model's probability, to show agreement/disagreement
        model_probs = []
        for name, m in _ALL_MODELS.items():
            try:
                p = float(m.predict_proba(Xp)[0, 1])
                model_probs.append({"model": name, "label": _MODEL_LABELS.get(name, name),
                                    "risk_percent": round(p * 100, 1)})
            except Exception:
                pass
        model_probs.sort(key=lambda d: d["risk_percent"], reverse=True)
        result["clinician"] = {
            "model_probabilities": model_probs,
            "all_contributions": contribs,       # full breakdown, not just top 5
            "subgroup": _subgroup(raw),
            "completed_inputs": raw,
            "engineered_shown": True,
        }

    return result


def feature_schema():
    return {
        "features": [
            {"name": f, "optional": f in OPTIONAL_FEATURES, **FEATURE_META[f]}
            for f in FEATURES
        ]
    }
