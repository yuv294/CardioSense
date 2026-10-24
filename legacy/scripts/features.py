"""Feature engineering shared by training and serving.

CRITICAL: this is the single place engineered features are defined. Both
preprocess.py (training) and the live API (serving) import add_engineered_
features from here, so a feature can never be computed one way in training and
another way at inference. That mismatch is a classic silent production bug;
sharing the function eliminates it.

Every engineered feature is a deterministic function of the 15 raw inputs and
encodes a known cardiovascular relationship -- no data-derived statistics are
baked in here (those belong in the fitted scaler/imputer), so this function is
safe to call on a single patient at serve time.
"""
import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered columns appended.

    Assumes the 15 raw columns are present. Missing raw values are tolerated:
    the arithmetic propagates NaN, which the downstream imputer handles, so we
    never invent a value here.
    """
    d = df.copy()

    # Pulse pressure: the gap between systolic and diastolic. A wide gap
    # reflects stiff arteries and independently predicts cardiac events.
    d["pulse_pressure"] = d["sysBP"] - d["diaBP"]

    # Mean arterial pressure: the average pressure over a cardiac cycle.
    d["map"] = d["diaBP"] + (d["sysBP"] - d["diaBP"]) / 3.0

    # Pressure burden scaled by age -- high BP matters more as you age.
    d["bp_age"] = d["sysBP"] * d["age"] / 100.0

    # Rough lifetime smoking load (a crude pack-years proxy from one visit).
    d["smoker_pack"] = d["cigsPerDay"] * d["age"] / 100.0

    # Cholesterol exposure scaled by age.
    d["chol_age"] = d["totChol"] * d["age"] / 1000.0

    # Compounding comorbidity: hypertension AND diabetes together is worse
    # than either alone.
    d["hyp_and_diab"] = ((d["prevalentHyp"] == 1) & (d["diabetes"] == 1)).astype(float)

    # Diabetic-range fasting glucose (ADA threshold 126 mg/dL).
    d["high_glucose"] = (d["glucose"] >= 126).astype(float)

    # Age enters most risk models on a log or squared scale; risk accelerates.
    d["age_sq"] = d["age"] ** 2

    return d


# ---------------------------------------------------------------------------
# Second feature layer (used by the improvement ladder, ladder.py)
#
# These three functions are kept apart from add_engineered_features so the
# original 23-feature model is untouched. Like the first layer they depend only
# on the same row's raw values, so they are safe to call on one patient.
# ---------------------------------------------------------------------------
from baseline_frs import COEF as _FRS_COEF  # published Framingham coefficients

STRUCTURE_FEATURES = ["ln_age", "ln_sbp", "ln_bmi", "male_x_ln_age", "male_x_ln_sbp",
                      "male_x_smoker", "male_x_diabetes", "treated_x_ln_sbp"]
# columns with at least 1% missing in the cohort (rule fixed before looking at results)
MISSING_FLAG_COLUMNS = ["glucose", "education", "BPMeds", "totChol"]
MISSING_FLAGS = [f"{c}_missing" for c in MISSING_FLAG_COLUMNS]


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copy the shape of the Framingham formula: log scales, separate effects for
    men and women (sex interactions) and a different blood-pressure term when the
    person is on blood-pressure medication."""
    d = df.copy()
    d["ln_age"] = np.log(d["age"])
    d["ln_sbp"] = np.log(d["sysBP"])
    d["ln_bmi"] = np.log(d["BMI"])
    d["male_x_ln_age"] = d["male"] * d["ln_age"]
    d["male_x_ln_sbp"] = d["male"] * d["ln_sbp"]
    d["male_x_smoker"] = d["male"] * d["currentSmoker"]
    d["male_x_diabetes"] = d["male"] * d["diabetes"]
    d["treated_x_ln_sbp"] = d["BPMeds"].fillna(0) * d["ln_sbp"]
    return d


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """1 when the original value was blank. Must be called BEFORE any imputation."""
    d = df.copy()
    for c in MISSING_FLAG_COLUMNS:
        d[f"{c}_missing"] = d[c].isna().astype(float)
    return d


def add_frs_logit(df: pd.DataFrame) -> pd.DataFrame:
    """The published Framingham risk of the person, on the log-odds scale. It is a
    fixed formula (no fitting), so it can be computed for any row; a blank age,
    BMI or blood pressure gives NaN and is imputed later like any other column."""
    d = df.copy()
    male = d["male"].fillna(0).values.astype(bool)
    treated = d["BPMeds"].fillna(0).values.astype(bool)
    smoker = d["currentSmoker"].fillna(0).values
    diabetes = d["diabetes"].fillna(0).values
    out = np.empty(len(d))
    for is_male, c in ((True, _FRS_COEF["men"]), (False, _FRS_COEF["women"])):
        m = male == is_male
        beta_sbp = np.where(treated[m], c["log_sbp_treated"], c["log_sbp_untreated"])
        lp = (c["log_age"] * np.log(d["age"].values[m]) + c["log_bmi"] * np.log(d["BMI"].values[m])
              + beta_sbp * np.log(d["sysBP"].values[m]) + c["smoker"] * smoker[m]
              + c["diabetes"] * diabetes[m])
        p = 1.0 - c["s0"] ** np.exp(lp - c["const"])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out[m] = np.log(p / (1 - p))
    d["frs_logit"] = out
    return d
