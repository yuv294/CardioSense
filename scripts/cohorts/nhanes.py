"""Cohort 2: NHANES 1999-2010 with NCHS linked mortality (follow-up to 2019).

Why a second cohort
-------------------
The Framingham score was derived in Framingham participants, so Framingham data
is its home ground. NHANES is an external test in every sense: a nationally
representative, multi-ethnic US sample examined 40 years later, in the statin
era, with laboratory markers that newer guidelines use (eGFR and urine albumin
in AHA PREVENT 2023, HbA1c, hs-CRP) and social factors (education, income).

Outcome: death from cardiovascular disease (ICD-10 I00-I09, I11, I13, I20-I51,
I60-I69; NCHS leading-cause groups 001 and 005) within 10 years of the exam.
Non-CVD death first is a competing event, labelled 0. Everyone alive at the
end of follow-up was followed for at least 9 years (the 2009-10 cycle has 9-10
years); they are labelled 0 rather than dropped, because dropping only the
survivors would inflate that cycle's event rate. People alive with under 9
years of follow-up would be dropped (there are none).

Neither published equation was built for CVD death, so their absolute risks are
on a different scale (they predict fatal + non-fatal events); discrimination
(AUROC) is still a fair comparison and calibration is compared after
recalibration too.

Inclusion: age 40-79, examined, eligible for mortality linkage, no self-reported
heart failure, CHD, angina, heart attack or stroke; total and HDL cholesterol,
systolic BP, smoking and diabetes status known (the inputs both published
equations need).
"""
import numpy as np
import pandas as pd

from baselines import frs_best, pce
from config import DATA_RAW

NAME = "nhanes"
TITLE = "NHANES 1999-2010, linked mortality to 2019"
OUTCOME = "Death from cardiovascular disease (heart disease or stroke) within 10 years"
OUTCOME_SHORT = "10-year CVD death"
SOURCE = "CDC National Health and Nutrition Examination Survey, 6 cycles, NCHS public-use linked mortality files"
HORIZON_MONTHS = 120
RAW = DATA_RAW / "nhanes"

FEATURES = ["male", "age", "race", "education", "pir", "smoking", "cigsPerDay", "BPMeds", "prevalentHyp",
            "diabetes", "famHistory", "sysBP", "diaBP", "totChol", "HDL", "BMI", "waist",
            "HbA1c", "eGFR", "uacr", "crp"]
OPTIONAL = ["education", "pir", "famHistory", "waist", "HbA1c", "eGFR", "uacr", "crp"]
MODIFIABLE = ["smoking", "cigsPerDay", "sysBP", "diaBP", "totChol", "HDL", "BMI", "waist", "HbA1c"]

FORM_GROUPS = [
    ("about", "About you", None),
    ("lifestyle", "Smoking", None),
    ("history", "Medical history", None),
    ("measure", "Measurements", None),
    ("labs", "Lab results (optional)", "Each one you add sharpens the estimate. Leave blank if unknown."),
]
FEATURE_META = {
    "male": {"label": "Sex", "kind": "binary", "options": {0: "Female", 1: "Male"}, "group": "about"},
    "age": {"label": "Age", "kind": "number", "unit": "years", "min": 40, "max": 79, "group": "about",
            "help": "The model covers ages 40 to 79."},
    "race": {"label": "Race / ethnicity", "kind": "select", "group": "about",
             "options": {3: "White", 4: "Black", 1: "Mexican American", 2: "Other Hispanic", 5: "Other / multiracial"}},
    "education": {"label": "Education", "kind": "select", "group": "about",
                  "options": {1: "Less than 9th grade", 2: "9-11th grade", 3: "High school", 4: "Some college",
                              5: "College graduate"}},
    "pir": {"label": "Household income ÷ poverty line", "kind": "number", "unit": "ratio", "min": 0, "max": 5,
            "step": 0.1, "group": "about", "help": "1 = at the poverty line, 5 = five times it or more."},
    "smoking": {"label": "Smoking", "kind": "select", "group": "lifestyle",
                "options": {0: "Never", 1: "Former", 2: "Current"}},
    "cigsPerDay": {"label": "Cigarettes per day", "kind": "number", "unit": "/day", "min": 0, "max": 80,
                   "group": "lifestyle"},
    "BPMeds": {"label": "Taking blood-pressure medicine", "kind": "binary", "options": {0: "No", 1: "Yes"}, "group": "history"},
    "prevalentHyp": {"label": "Diagnosed high blood pressure", "kind": "binary", "options": {0: "No", 1: "Yes"}, "group": "history"},
    "diabetes": {"label": "Diabetes", "kind": "binary", "options": {0: "No", 1: "Yes"}, "group": "history"},
    "famHistory": {"label": "Parent or sibling had a heart attack before 50", "kind": "binary",
                   "options": {0: "No", 1: "Yes"}, "group": "history"},
    "sysBP": {"label": "Systolic BP", "kind": "number", "unit": "mm Hg", "min": 80, "max": 240, "group": "measure",
              "help": "The top number of a reading such as 120/80."},
    "diaBP": {"label": "Diastolic BP", "kind": "number", "unit": "mm Hg", "min": 30, "max": 140, "group": "measure"},
    "totChol": {"label": "Total cholesterol", "kind": "number", "unit": "mg/dL", "min": 80, "max": 450, "group": "measure"},
    "HDL": {"label": "HDL cholesterol", "kind": "number", "unit": "mg/dL", "min": 10, "max": 150, "group": "measure",
            "help": "The 'good' cholesterol."},
    "BMI": {"label": "Body mass index", "kind": "number", "unit": "kg/m²", "min": 15, "max": 70, "group": "measure"},
    "waist": {"label": "Waist", "kind": "number", "unit": "cm", "min": 50, "max": 180, "group": "measure"},
    "HbA1c": {"label": "HbA1c", "kind": "number", "unit": "%", "min": 3.5, "max": 15, "step": 0.1, "group": "labs",
              "help": "Average blood sugar over 3 months."},
    "eGFR": {"label": "eGFR (kidney function)", "kind": "number", "unit": "mL/min", "min": 5, "max": 150, "group": "labs"},
    "uacr": {"label": "Urine albumin / creatinine", "kind": "number", "unit": "mg/g", "min": 0, "max": 3000,
             "step": 0.1, "group": "labs", "help": "Under 30 is normal."},
    "crp": {"label": "C-reactive protein (hs-CRP)", "kind": "number", "unit": "mg/L", "min": 0, "max": 100,
            "step": 0.1, "group": "labs", "help": "A marker of inflammation."},
}
FACTOR_OF = {"age": "Age", "male": "Sex", "race": "Background", "education": "Social factors", "pir": "Social factors",
             "smoking": "Smoking", "cigsPerDay": "Smoking", "BPMeds": "Blood pressure", "prevalentHyp": "Blood pressure",
             "sysBP": "Blood pressure", "diaBP": "Blood pressure", "diabetes": "Blood sugar", "HbA1c": "Blood sugar",
             "famHistory": "Family history", "totChol": "Cholesterol", "HDL": "Cholesterol", "BMI": "Body weight",
             "waist": "Body weight", "eGFR": "Kidney health", "uacr": "Kidney health", "crp": "Inflammation"}

# Risk bands for 10-year CVD death (ESC SCORE convention) and the decision-curve range.
BANDS = [(0.01, "low", "Lower risk"), (0.05, "intermediate", "Moderate"), (1.01, "high", "Higher risk")]
BAND_TEXT = {"low": "Fewer than 1 in 100 people with this profile die of heart disease or stroke within ten years.",
             "intermediate": "Between 1 and 5 in 100. Worth reviewing the modifiable factors with a doctor.",
             "high": "5 in 100 or more, the level at which guidelines advise active risk-factor treatment."}
DCA_RANGE = (0.005, 0.15)

STRATA = ["cycle"]
CANDIDATES = {}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise only (safe for a single patient at serve time). Missing stays NaN."""
    d = df[FEATURES].astype(float).copy()
    d["race_black"] = (d["race"] == 4).astype(float)
    d["race_hisp"] = d["race"].isin([1, 2]).astype(float)
    d["race_other"] = (d["race"] == 5).astype(float)
    d["currentSmoker"] = (d["smoking"] == 2).astype(float)
    d["formerSmoker"] = (d["smoking"] == 1).astype(float)
    d["ln_age"] = np.log(d["age"])
    d["ln_sbp"] = np.log(d["sysBP"])
    d["ln_tc"] = np.log(d["totChol"])
    d["ln_hdl"] = np.log(d["HDL"])
    d["ln_bmi"] = np.log(d["BMI"])
    d["ln_cigs"] = np.log1p(d["cigsPerDay"].fillna(0))
    d["treated_sbp"] = d["BPMeds"].fillna(0) * d["ln_sbp"]
    d["male_x_ln_age"] = d["male"] * d["ln_age"]
    d["age_x_smoker"] = d["ln_age"] * d["currentSmoker"]
    d["tc_hdl_ratio"] = d["totChol"] / d["HDL"]
    # kidney: risk rises below ~60 mL/min (PREVENT uses a spline with a knot at 60)
    d["egfr_below60"] = np.clip(60 - d["eGFR"], 0, None)
    d["egfr_above60"] = np.clip(d["eGFR"] - 60, 0, None)
    d["ln_uacr"] = np.log1p(d["uacr"])
    d["ln_crp"] = np.log1p(d["crp"])
    d["pce_logit"] = _logit(pce(d.fillna({"totChol": 200.0, "HDL": 50.0, "sysBP": 125.0})))
    return d


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


MONOTONE = {"age": 1, "ln_age": 1, "sysBP": 1, "ln_sbp": 1, "HDL": -1, "ln_hdl": -1, "cigsPerDay": 1, "ln_cigs": 1,
            "diabetes": 1, "currentSmoker": 1, "HbA1c": 1, "uacr": 1, "ln_uacr": 1, "egfr_below60": 1,
            "pce_logit": 1, "prevalentHyp": 1, "famHistory": 1}
REFIT_COLS = ["ln_age", "male", "male_x_ln_age", "ln_sbp", "treated_sbp", "currentSmoker", "age_x_smoker",
              "diabetes", "ln_tc", "ln_hdl", "race_black"]
SPLINE_COLS = ["age", "sysBP", "BMI", "HbA1c"]
BASELINES = {"pce": pce, "frs_best": frs_best}
PRIMARY_BASELINE = "pce"

# ---------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------
CYCLES = {1999: "", 2001: "_B", 2003: "_C", 2005: "_D", 2007: "_E", 2009: "_F"}
LAB_FILES = {1999: ["Lab13", "Lab10", "Lab18", "Lab11", "Lab16"], 2001: ["L13_B", "L10_B", "L40_B", "L11_B", "L16_B"],
             2003: ["L13_C", "L10_C", "L40_C", "L11_C", "L16_C"]}


def _xpt(path):
    d = pd.read_sas(path, format="xport")
    num = d.select_dtypes("number").columns
    tiny = d[num].abs() < 1e-20                          # XPT stores 0 as a denormal like 5.4e-79
    d[num] = d[num].mask(tiny, 0.0)                       # (NaN compares False, so it stays NaN)
    d["SEQN"] = d["SEQN"].astype(int)
    return d.set_index("SEQN")


def _first(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return pd.Series(np.nan, index=d.index)


def _mortality(year):
    rows = []
    for line in (RAW / "mort" / f"{year}.dat").read_text().splitlines():
        if len(line) < 48:
            line = line.ljust(48)
        num = lambda s: float(s) if s.strip() not in ("", ".") else np.nan
        rows.append({"SEQN": int(line[0:14]), "elig": num(line[14]), "mortstat": num(line[15]),
                     "ucod": num(line[16:19]), "permth_exm": num(line[45:48])})
    return pd.DataFrame(rows).set_index("SEQN")


def _ckd_epi_2021(scr, age, male):
    k = np.where(male == 1, 0.9, 0.7)
    a = np.where(male == 1, -0.302, -0.241)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(male == 1, 1.0, 1.012)


def _cycle(year):
    suf = CYCLES[year]
    f = lambda n: _xpt(RAW / str(year) / f"{n}.xpt")
    parts = [f(q + suf) for q in ("DEMO", "BPX", "BMX", "BPQ", "DIQ", "SMQ", "MCQ")]
    labs = LAB_FILES.get(year) or [l + suf for l in ("TCHOL", "HDL", "GHB", "BIOPRO", "CRP", "ALB_CR")]
    parts += [f(n) for n in labs]
    d = parts[0]
    for p in parts[1:]:
        d = d.join(p[[c for c in p.columns if c not in d.columns]], how="left")
    d = d.join(_mortality(year), how="left")

    o = pd.DataFrame(index=d.index)
    o["cycle"] = year
    o["male"] = (d["RIAGENDR"] == 1).astype(int)
    o["age"] = d["RIDAGEYR"]
    o["race"] = d["RIDRETH1"]
    o["education"] = d["DMDEDUC2"].where(d["DMDEDUC2"].between(1, 5))
    o["pir"] = d["INDFMPIR"]
    ever, now = d["SMQ020"], d["SMQ040"]
    o["smoking"] = np.select([ever == 2, now.isin([1, 2]), now == 3], [0, 2, 1], default=np.nan)
    cigs = _first(d, "SMD650", "SMD070").where(lambda s: s < 777)
    o["cigsPerDay"] = np.where(o["smoking"] == 2, cigs, np.where(o["smoking"].notna(), 0, np.nan))
    o["prevalentHyp"] = np.select([d["BPQ020"] == 1, d["BPQ020"] == 2], [1, 0], default=np.nan)
    o["BPMeds"] = (d.get("BPQ050A") == 1).astype(int)
    dm_meds = (_first(d, "DIQ050") == 1) | (_first(d, "DIQ070") == 1)
    o["diabetes"] = np.select([(d["DIQ010"] == 1) | dm_meds, d["DIQ010"].isin([2, 3])], [1, 0], default=np.nan)
    o["famHistory"] = np.select([_first(d, "MCQ300A") == 1, _first(d, "MCQ300A") == 2], [1, 0], default=np.nan)
    sbp = d[[c for c in ("BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4") if c in d]].replace(0, np.nan)
    dbp = d[[c for c in ("BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4") if c in d]].replace(0, np.nan)
    o["sysBP"], o["diaBP"] = sbp.mean(axis=1), dbp.mean(axis=1)
    o["totChol"] = d["LBXTC"]
    o["HDL"] = _first(d, "LBDHDD", "LBXHDD", "LBDHDL")
    o["BMI"], o["waist"] = d["BMXBMI"], d["BMXWAIST"]
    o["HbA1c"] = d["LBXGH"]
    scr = _first(d, "LBXSCR", "LBDSCR")
    # NHANES-recommended recalibration of serum creatinine to the standardised assay
    if year == 1999:
        scr = 1.013 * scr + 0.147
    elif year == 2005:
        scr = -0.016 + 0.978 * scr
    o["eGFR"] = _ckd_epi_2021(scr, o["age"], o["male"])
    o["uacr"] = d["URXUMA"] / d["URXUCR"] * 100.0               # (mg/L) / (mg/dL) -> mg/g
    o["crp"] = d["LBXCRP"] * 10.0                               # mg/dL -> mg/L
    prior = pd.concat([d.get(c) == 1 for c in ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F")], axis=1).any(axis=1)
    o["prior_cvd"] = prior
    o["elig"], o["mortstat"], o["ucod"], o["months"] = d["elig"], d["mortstat"], d["ucod"], d["permth_exm"]
    return o


def build():
    d = pd.concat([_cycle(y) for y in CYCLES])
    log = {"participants_all_ages": int(len(d))}
    d = d[d["age"].between(40, 79)]
    log["aged_40_79"] = int(len(d))
    d = d[(d["elig"] == 1) & d["months"].notna()]
    log["eligible_for_mortality_linkage_and_examined"] = int(len(d))
    d = d[~d["prior_cvd"]]
    log["no_prior_cvd"] = int(len(d))
    d = d.dropna(subset=["sysBP", "totChol", "HDL", "smoking", "diabetes"])
    log["core_inputs_known"] = int(len(d))

    cvd_death = (d["mortstat"] == 1) & d["ucod"].isin([1, 5]) & (d["months"] <= HORIZON_MONTHS)
    alive_short = (d["mortstat"] == 0) & (d["months"] < 108)
    log["competing_deaths_labelled_0"] = int(((d["mortstat"] == 1) & ~d["ucod"].isin([1, 5])
                                              & (d["months"] <= HORIZON_MONTHS)).sum())
    log["dropped_alive_with_under_9y_followup"] = int(alive_short.sum())
    d["y"] = cvd_death.astype(int)
    d = d[~alive_short]
    d["pid"] = d.index.astype(int)
    out = d[["pid", "y"] + STRATA + FEATURES].reset_index(drop=True)
    out["prevalentHyp"] = out["prevalentHyp"].fillna(out["BPMeds"])
    log["by_cycle"] = {int(k): {"rows": int(len(v)), "event_rate": round(float(v["y"].mean()), 4)}
                       for k, v in out.groupby("cycle")}
    return out, log
