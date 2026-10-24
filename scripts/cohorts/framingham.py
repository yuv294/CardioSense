"""Cohort 1: Framingham Heart Study teaching dataset (NHLBI / BioLINCC).

Why this replaces the Kaggle CSV used in v1
-------------------------------------------
The Kaggle `framingham.csv` is a one-visit extract (exam 1) of this cohort with
a pre-computed label. The teaching dataset (4,434 participants, three exams
about six years apart, 24 years of adjudicated follow-up) allows:

* Outcome rebuilt from event times: CHD (angina, MI, coronary insufficiency or
  CHD death) within 10 years of the visit.
* Primary prevention only: visits with prior CHD or stroke are removed.
* Landmarking (van Houwelingen 2007): every eligible visit is a starting point,
  so a person contributes up to three rows (rows of one person always stay
  together in splits and CV folds). This doubles the events available.
* HDL cholesterol (exam 3), so the laboratory Framingham score can be used.

Censoring: death from another cause before a CHD event is a competing event and
labelled 0. Rows censored for any other reason before 10 years are dropped.

Inputs were chosen by a grouped-CV ablation on the training participants
(reports/framingham/ablation.json): previous-visit values, glucose, heart rate
and education added nothing or made CV log-loss worse, so they are not asked for.

Note: the Framingham score was DERIVED in Framingham participants. Beating it
here is the hardest possible test; see the NHANES cohort for an external one.
"""
import numpy as np
import pandas as pd
import pyreadr

from baselines import frs_best, frs_lab, frs_office
from config import DATA_RAW

NAME = "framingham"
TITLE = "Framingham Heart Study (teaching cohort)"
OUTCOME = "First coronary heart disease event (angina, heart attack, coronary insufficiency or CHD death) within 10 years"
OUTCOME_SHORT = "10-year CHD event"
SOURCE = "NHLBI Framingham teaching dataset, 4,434 participants, exams 1956-1968, 24-year follow-up"
HORIZON_DAYS = 3652

FEATURES = ["male", "age", "sysBP", "diaBP", "BPMeds", "prevalentHyp", "currentSmoker", "cigsPerDay",
            "totChol", "HDL", "BMI", "diabetes"]
OPTIONAL = ["HDL"]
MODIFIABLE = ["currentSmoker", "cigsPerDay", "sysBP", "diaBP", "totChol", "HDL", "BMI"]

FORM_GROUPS = [
    ("about", "About you", None),
    ("lifestyle", "Smoking", None),
    ("history", "Medical history", None),
    ("measure", "Measurements", "From your latest check-up. Leave HDL blank if you don't know it."),
]
FEATURE_META = {
    "male": {"label": "Sex", "kind": "binary", "options": {0: "Female", 1: "Male"}, "group": "about"},
    "age": {"label": "Age", "kind": "number", "unit": "years", "min": 32, "max": 81, "group": "about",
            "help": "The study cohort was aged 32 to 81."},
    "currentSmoker": {"label": "Current smoker", "kind": "binary", "options": {0: "No", 1: "Yes"}, "group": "lifestyle"},
    "cigsPerDay": {"label": "Cigarettes per day", "kind": "number", "unit": "/day", "min": 0, "max": 80, "group": "lifestyle"},
    "BPMeds": {"label": "Taking blood-pressure medicine", "kind": "binary", "options": {0: "No", 1: "Yes"}, "group": "history"},
    "prevalentHyp": {"label": "Diagnosed high blood pressure", "kind": "binary", "options": {0: "No", 1: "Yes"}, "group": "history"},
    "diabetes": {"label": "Diabetes", "kind": "binary", "options": {0: "No", 1: "Yes"}, "group": "history"},
    "sysBP": {"label": "Systolic BP", "kind": "number", "unit": "mm Hg", "min": 80, "max": 260, "group": "measure",
              "help": "The top number of a reading such as 120/80."},
    "diaBP": {"label": "Diastolic BP", "kind": "number", "unit": "mm Hg", "min": 40, "max": 150, "group": "measure"},
    "totChol": {"label": "Total cholesterol", "kind": "number", "unit": "mg/dL", "min": 100, "max": 500, "group": "measure"},
    "HDL": {"label": "HDL cholesterol", "kind": "number", "unit": "mg/dL", "min": 10, "max": 150, "group": "measure",
            "help": "The 'good' cholesterol."},
    "BMI": {"label": "Body mass index", "kind": "number", "unit": "kg/m²", "min": 15, "max": 60, "group": "measure"},
}
FACTOR_OF = {"age": "Age", "male": "Sex", "sysBP": "Blood pressure", "diaBP": "Blood pressure",
             "BPMeds": "Blood pressure", "prevalentHyp": "Blood pressure", "currentSmoker": "Smoking",
             "cigsPerDay": "Smoking", "totChol": "Cholesterol", "HDL": "Cholesterol", "BMI": "Body weight",
             "diabetes": "Diabetes"}

# Risk bands (ATP III convention for 10-year CHD risk) and the decision-curve range.
BANDS = [(0.10, "low", "Lower risk"), (0.20, "intermediate", "Intermediate"), (1.01, "high", "Higher risk")]
BAND_TEXT = {"low": "Fewer than 1 in 10 people with this profile have a CHD event within ten years.",
             "intermediate": "Between 1 in 10 and 1 in 5. Worth reviewing the modifiable factors with a doctor.",
             "high": "1 in 5 or more. Worth a conversation with a clinician soon."}
DCA_RANGE = (0.02, 0.40)

STRATA = ["exam"]
# Inputs considered and rejected by the ablation (kept in the analysis table so
# ablation.py can re-test them). Each group is added to the final inputs in turn.
CANDIDATES = {"previous visit": ["prev_sysBP", "prev_totChol", "prev_BMI", "prev_smoker"],
              "glucose": ["glucose"], "heart rate": ["heartRate"], "education": ["education"]}
BASELINES = {"frs_best": frs_best, "frs_office": frs_office, "frs_lab": frs_lab}
PRIMARY_BASELINE = "frs_best"


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise only (safe for a single patient at serve time). Missing stays NaN."""
    d = df[FEATURES].astype(float).copy()
    d["ln_age"] = np.log(d["age"])
    d["ln_sbp"] = np.log(d["sysBP"])
    d["ln_tc"] = np.log(d["totChol"])
    d["ln_hdl"] = np.log(d["HDL"])
    d["ln_bmi"] = np.log(d["BMI"])
    d["ln_cigs"] = np.log1p(d["cigsPerDay"].fillna(0))
    d["treated_sbp"] = d["BPMeds"].fillna(0) * d["ln_sbp"]
    d["male_x_ln_age"] = d["male"] * d["ln_age"]
    d["male_x_smoker"] = d["male"] * d["currentSmoker"]
    d["pulse_pressure"] = d["sysBP"] - d["diaBP"]
    d["tc_hdl_ratio"] = d["totChol"] / d["HDL"]
    d["frs_office_logit"] = _logit(frs_office(d))
    return d


MONOTONE = {"age": 1, "ln_age": 1, "sysBP": 1, "ln_sbp": 1, "totChol": 1, "ln_tc": 1, "HDL": -1, "ln_hdl": -1,
            "cigsPerDay": 1, "ln_cigs": 1, "diabetes": 1, "tc_hdl_ratio": 1, "frs_office_logit": 1, "prevalentHyp": 1}
REFIT_COLS = ["ln_age", "male", "male_x_ln_age", "ln_sbp", "treated_sbp", "currentSmoker", "male_x_smoker",
              "diabetes", "ln_tc", "ln_hdl", "ln_bmi"]
SPLINE_COLS = ["age", "sysBP", "totChol", "BMI"]

RENAME = {"SEX": "sex", "AGE": "age", "SYSBP": "sysBP", "DIABP": "diaBP", "BPMEDS": "BPMeds",
          "CURSMOKE": "currentSmoker", "CIGPDAY": "cigsPerDay", "TOTCHOL": "totChol", "HDLC": "HDL",
          "BMI": "BMI", "DIABETES": "diabetes", "PREVHYP": "prevalentHyp", "HEARTRTE": "heartRate",
          "GLUCOSE": "glucose", "educ": "education"}


def build():
    raw = next(iter(pyreadr.read_r(str(DATA_RAW / "frmgham_teaching.rda")).values()))
    d = raw.sort_values(["RANDID", "PERIOD"]).rename(columns=RENAME).reset_index(drop=True)
    d["male"] = (d["sex"] == 1).astype(int)
    g = d.groupby("RANDID")
    for c in ("sysBP", "totChol", "BMI"):
        d[f"prev_{c}"] = g[c].shift(1)
    d["prev_smoker"] = g["currentSmoker"].shift(1)

    log = {"visits_total": int(len(d)), "participants_total": int(d["RANDID"].nunique())}
    d = d[(d["PREVCHD"] == 0) & (d["PREVSTRK"] == 0)].copy()
    log["visits_after_excluding_prior_chd_or_stroke"] = int(len(d))

    t_left = d["TIMECHD"] - d["TIME"]
    event = (d["ANYCHD"] == 1) & (t_left <= HORIZON_DAYS) & (t_left > 0)
    short = (~event) & (t_left < HORIZON_DAYS)
    died = short & (d["DEATH"] == 1) & (d["TIMEDTH"] <= d["TIMECHD"])
    drop = short & ~died
    log["competing_deaths_labelled_0"] = int(died.sum())
    log["dropped_censored_before_10y"] = int(drop.sum())

    d["y"] = event.astype(int)
    d["pid"] = d["RANDID"].astype(int)
    d["exam"] = d["PERIOD"].astype(int)
    d = d[~drop]
    extra = [c for cols in CANDIDATES.values() for c in cols]
    out = d[["pid", "y"] + STRATA + FEATURES + extra].reset_index(drop=True)
    log["by_exam"] = {int(k): {"rows": int(len(v)), "event_rate": round(float(v["y"].mean()), 4)}
                      for k, v in out.groupby("exam")}
    return out, log
