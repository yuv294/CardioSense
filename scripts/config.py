"""Shared configuration for the CardioSense v2 pipeline.

Paths, seed and split settings. Everything cohort-specific (inputs, outcome,
baselines) lives in scripts/cohorts/<name>.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

RANDOM_STATE = 42
TEST_FRACTION = 0.20      # share of PARTICIPANTS locked away for the final test (opened once, by evaluate.py)
CV_FOLDS = 5
TARGET = "y"
GROUP = "pid"             # participant id: all rows of one person stay on one side of every split

COHORTS = ["framingham", "nhanes"]


def dirs(cohort: str):
    out = {k: base / cohort for k, base in
           (("data", DATA_PROCESSED), ("art", ARTIFACTS), ("rep", REPORTS))}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    (out["rep"] / "figures").mkdir(exist_ok=True)
    return out
