"""Shared configuration for the CardioSense pipeline.

Single source of truth for paths, the feature list, the target, and the
random seed. Every script imports from here so the split, the models, the
evaluation, and the served API all agree on column order and seeding.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "framingham.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

for _d in (DATA_PROCESSED, ARTIFACTS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# One seed used everywhere: the split, the CV folds, and every model that
# accepts a random_state. Fixing it means a teammate who reruns the pipeline
# gets the same numbers.
RANDOM_STATE = 42

# Fraction of data held out as the locked test set. Opened exactly once, at
# the very end, by evaluate.py. No tuning decision may look at it.
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Features and target
# ---------------------------------------------------------------------------
TARGET = "TenYearCHD"

# The 15 predictors, grouped by how a person would supply them. Every one of
# these is something a patient knows or a single clinic visit measures -- no
# stress-test or angiogram results, which was the flaw in the old UCI build.
FEATURES = [
    # demographics
    "male", "age", "education",
    # behaviour
    "currentSmoker", "cigsPerDay",
    # medical history
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    # measurements (clinic visit)
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]

# Human-readable labels, units, and input hints -- consumed by the API and
# the frontend form so the UI and the model never disagree about a field.
FEATURE_META = {
    "male":            {"label": "Sex",                      "kind": "binary", "options": {0: "Female", 1: "Male"}},
    "age":             {"label": "Age",                      "kind": "number", "unit": "years",  "min": 20, "max": 100},
    "education":       {"label": "Education level",          "kind": "ordinal","unit": "1-4",    "min": 1,  "max": 4},
    "currentSmoker":   {"label": "Current smoker",           "kind": "binary", "options": {0: "No", 1: "Yes"}},
    "cigsPerDay":      {"label": "Cigarettes per day",       "kind": "number", "unit": "count",  "min": 0,  "max": 90},
    "BPMeds":          {"label": "On blood-pressure meds",   "kind": "binary", "options": {0: "No", 1: "Yes"}},
    "prevalentStroke": {"label": "Prior stroke",             "kind": "binary", "options": {0: "No", 1: "Yes"}},
    "prevalentHyp":    {"label": "Hypertensive",             "kind": "binary", "options": {0: "No", 1: "Yes"}},
    "diabetes":        {"label": "Diabetic",                 "kind": "binary", "options": {0: "No", 1: "Yes"}},
    "totChol":         {"label": "Total cholesterol",        "kind": "number", "unit": "mg/dL",  "min": 100,"max": 600},
    "sysBP":           {"label": "Systolic BP",              "kind": "number", "unit": "mm Hg",  "min": 80, "max": 300},
    "diaBP":           {"label": "Diastolic BP",             "kind": "number", "unit": "mm Hg",  "min": 40, "max": 150},
    "BMI":             {"label": "Body mass index",          "kind": "number", "unit": "kg/m^2", "min": 15, "max": 60},
    "heartRate":       {"label": "Resting heart rate",       "kind": "number", "unit": "bpm",    "min": 40, "max": 150},
    "glucose":         {"label": "Fasting glucose",          "kind": "number", "unit": "mg/dL",  "min": 40, "max": 400},
}

# Of the 15, these four are optional on the form. They are the most-missing
# columns in the data (glucose, education) or ones a person may not know
# offhand (cigsPerDay for non-smokers is simply 0). Anything left blank is
# filled with the TRAINING-set median/mode and reported back as an assumed
# field, so the user always sees what was guessed.
OPTIONAL_FEATURES = ["education", "glucose", "totChol", "cigsPerDay"]

# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------
# Derived from the 15 raw inputs by scripts/features.py. These encode clinical
# relationships the raw columns only imply -- the kind of domain knowledge that
# actually moves the metrics, as opposed to throwing more trees at the problem.
# Training and serving BOTH call the same add_engineered_features(), so the
# columns can never drift between the two.
ENGINEERED_FEATURES = [
    "pulse_pressure",   # sysBP - diaBP: arterial stiffness, a strong CVD signal
    "map",              # mean arterial pressure: diaBP + (pulse_pressure/3)
    "bp_age",           # sysBP * age / 100: pressure burden scaled by age
    "smoker_pack",      # cigsPerDay * age / 100: rough lifetime smoking load
    "chol_age",         # totChol * age / 1000: exposure scaled by age
    "hyp_and_diab",     # prevalentHyp AND diabetes: compounding comorbidity
    "high_glucose",     # glucose >= 126: diabetic-range fasting glucose flag
    "age_sq",           # age^2: risk accelerates non-linearly with age
]
