"""Step 5: per-prediction explanations the app can show the user.

The served model is a calibrated logistic regression. For a linear model the
honest, exact explanation is the signed contribution of each standardised
feature: coefficient * (standardised feature value). That is what SHAP reduces
to for a linear model anyway, with none of the approximation, so we compute it
directly and avoid adding shap as a deployment dependency.

We pull the logistic-regression coefficients out of the calibrated wrapper and
save them, plus the preprocessing mean/scale, so the backend can compute each
input's push on the risk -- "smoking added most to your risk, healthy BP
reduced it" -- without loading SHAP or the full training stack at serve time.

Output: artifacts/explainer.json
"""
import json
import numpy as np
import joblib

from config import ARTIFACTS

def main():
    served = joblib.load(ARTIFACTS / "model_served.joblib")
    proc_cols = json.loads((ARTIFACTS / "processed_columns.json").read_text())
    pre = joblib.load(ARTIFACTS / "preprocessor.joblib")

    # Dig the base logistic-regression estimators out of the calibrated wrapper.
    # CalibratedClassifierCV holds one fitted sub-estimator per CV fold; their
    # coefficients are near-identical, so we average them for a single stable
    # linear explanation.
    coefs, intercepts = [], []
    if hasattr(served, "calibrated_classifiers_"):
        for cc in served.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            coefs.append(est.coef_.ravel())
            intercepts.append(est.intercept_.ravel()[0])
        coef = np.mean(coefs, axis=0)
        intercept = float(np.mean(intercepts))
    else:
        coef = served.coef_.ravel()
        intercept = float(served.intercept_.ravel()[0])

    # Standardisation stats from the ColumnTransformer's continuous pipe.
    # Binary columns were only imputed, not scaled, so mean=0 scale=1 for them.
    cont_cols = pre.transformers_[0][2]
    scaler = pre.named_transformers_["cont"].named_steps["scale"]
    mean_map = {c: 0.0 for c in proc_cols}
    scale_map = {c: 1.0 for c in proc_cols}
    for c, m, s in zip(cont_cols, scaler.mean_, scaler.scale_):
        mean_map[c] = float(m)
        scale_map[c] = float(s)

    explainer = {
        "processed_columns": proc_cols,
        "coef": {c: float(w) for c, w in zip(proc_cols, coef)},
        "intercept": intercept,
        "mean": mean_map,
        "scale": scale_map,
        "note": ("contribution_i = coef_i * (x_i - mean_i)/scale_i; "
                 "positive pushes risk up. Sum + intercept = log-odds before "
                 "isotonic calibration; use only for ranking feature influence."),
    }
    (ARTIFACTS / "explainer.json").write_text(json.dumps(explainer, indent=2))
    print("Saved artifacts/explainer.json")
    # show the global ranking for a sanity check
    ranked = sorted(explainer["coef"].items(), key=lambda kv: abs(kv[1]), reverse=True)
    print("Global feature influence (|coef| on standardised inputs):")
    for c, w in ranked:
        print(f"  {c:16s} {w:+.3f}")


if __name__ == "__main__":
    main()
