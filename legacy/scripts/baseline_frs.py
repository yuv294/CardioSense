"""Step 2 of the pipeline: the rule-based baseline to beat.

This implements the published Framingham "General Cardiovascular Disease"
10-year risk function -- the simple, office-based (BMI) version from
D'Agostino et al., Circulation 2008 -- using the exact sex-specific
coefficients published by the Framingham Heart Study.

Source of coefficients (verified):
    Framingham Heart Study, "Cardiovascular Disease (10-year risk)",
    simple/office (BMI) model.

                               Men          Women
    Log Age                    3.11296      2.72107
    Log BMI                    0.79277      0.51125
    Log SBP (untreated)        1.85508      2.81291
    Log SBP (treated)          1.92672      2.88267
    Smoking                    0.70953      0.61868
    Diabetes                   0.53160      0.77763
    Baseline survival S0(10)   0.88431      0.94833
    Constant (subtracted)     23.9388      26.0145

    10-year risk = 1 - S0 ** exp( sum(beta*X) - constant )

Why this matters for the project
--------------------------------
"Our model gets AUROC 0.7x" is meaningless alone. The honest question is
whether three ML models can beat the calculator a clinician already has. This
file produces that calculator's predictions on the SAME locked test set the
ML models are judged on, so the comparison is apples to apples.

Note on scope: this office model omits total cholesterol (it is the
non-laboratory version). That is deliberate -- it is the variant whose inputs
our dataset fully supports. The lab version needs HDL, which Framingham's
teaching CSV does not contain.
"""
import numpy as np
import pandas as pd

from config import DATA_PROCESSED

COEF = {
    "men": dict(log_age=3.11296, log_bmi=0.79277, log_sbp_untreated=1.85508,
                log_sbp_treated=1.92672, smoker=0.70953, diabetes=0.53160,
                s0=0.88431, const=23.9388),
    "women": dict(log_age=2.72107, log_bmi=0.51125, log_sbp_untreated=2.81291,
                  log_sbp_treated=2.88267, smoker=0.61868, diabetes=0.77763,
                  s0=0.94833, const=26.0145),
}


def frs_risk_row(age, bmi, sbp, treated, smoker, diabetes, is_male):
    """10-year CVD risk for one person, as a probability in [0, 1]."""
    c = COEF["men"] if is_male else COEF["women"]
    sbp_beta = c["log_sbp_treated"] if treated else c["log_sbp_untreated"]
    lp = (c["log_age"] * np.log(age)
          + c["log_bmi"] * np.log(bmi)
          + sbp_beta * np.log(sbp)
          + c["smoker"] * (1 if smoker else 0)
          + c["diabetes"] * (1 if diabetes else 0))
    return 1.0 - c["s0"] ** np.exp(lp - c["const"])


def frs_risk_frame(df):
    """Vectorised risk for a dataframe with the raw Framingham columns.

    Missing inputs to the formula are filled with that column's median so the
    baseline can score every test row (it is a fixed formula, not a learned
    model, so there is no train/test leak concern here).
    """
    d = df.copy()
    for col in ["age", "BMI", "sysBP"]:
        d[col] = d[col].fillna(d[col].median())
    for col in ["BPMeds", "currentSmoker", "diabetes", "male"]:
        d[col] = d[col].fillna(0)

    out = np.empty(len(d), dtype=float)
    for i, (_, r) in enumerate(d.iterrows()):
        out[i] = frs_risk_row(
            age=r["age"], bmi=r["BMI"], sbp=r["sysBP"],
            treated=bool(r["BPMeds"]), smoker=bool(r["currentSmoker"]),
            diabetes=bool(r["diabetes"]), is_male=bool(r["male"]),
        )
    return out


def main():
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    risk = frs_risk_frame(X_test)
    np.save(DATA_PROCESSED / "frs_test_risk.npy", risk)
    print(f"Framingham baseline scored {len(risk)} test rows.")
    print(f"  mean predicted 10-yr risk: {risk.mean():.1%}")
    print(f"  min / max: {risk.min():.1%} / {risk.max():.1%}")
    # quick sanity check against a known reference profile:
    # 50yo non-smoking non-diabetic man, BMI 25, untreated SBP 125 -> about 9% (general CVD, D'Agostino 2008)
    demo = frs_risk_row(50, 25, 125, False, False, False, True)
    print(f"  sanity (50M, BMI25, SBP125, healthy): {demo:.1%}  (expected about 9%)")


if __name__ == "__main__":
    main()
