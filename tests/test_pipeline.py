"""Fast checks of the pieces most likely to break silently.

    pip install pytest && pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

import cohorts  # noqa: E402
from baselines import frs_office, frs_lab, pce  # noqa: E402
from metrics import calibration_stats, cluster_indices  # noqa: E402


def test_frs_office_is_plausible_and_monotone():
    # risk must be plausible and rise with every adverse factor
    base = pd.DataFrame([{"male": 1, "age": 55, "sysBP": 130, "BPMeds": 0, "currentSmoker": 0, "diabetes": 0, "BMI": 26}])
    worse = base.assign(currentSmoker=1, sysBP=160, diabetes=1)
    assert 0.03 < frs_office(base)[0] < 0.3
    assert frs_office(worse)[0] > frs_office(base)[0]


def test_frs_lab_needs_hdl():
    d = pd.DataFrame([{"male": 0, "age": 60, "sysBP": 140, "BPMeds": 0, "currentSmoker": 0, "diabetes": 0,
                       "totChol": 220, "HDL": np.nan}])
    assert np.isnan(frs_lab(d)[0])


def test_pce_reference_value():
    # ACC/AHA calculator reference: 55-year-old white woman, TC 213, HDL 50, SBP 120 untreated,
    # non-smoker, no diabetes -> 2.1% (Goff 2014 Table A worked example)
    d = pd.DataFrame([{"male": 0, "race_black": 0, "age": 55, "totChol": 213, "HDL": 50, "sysBP": 120,
                       "BPMeds": 0, "currentSmoker": 0, "diabetes": 0}])
    assert abs(pce(d)[0] - 0.021) < 0.003
    # same profile, white man -> 5.3%
    assert abs(pce(d.assign(male=1))[0] - 0.053) < 0.004


@pytest.mark.parametrize("name", ["framingham", "nhanes"])
def test_features_are_row_wise(name):
    """add_features of one row must equal that row computed inside a batch (no cross-row leakage)."""
    c = cohorts.get(name)
    p = ROOT / "data" / "processed" / name / "train.csv"
    if not p.exists():
        pytest.skip("run the pipeline first")
    df = pd.read_csv(p).head(50)
    batch = c.add_features(df)
    single = c.add_features(df.iloc[[17]])
    pd.testing.assert_frame_equal(batch.iloc[[17]].reset_index(drop=True), single.reset_index(drop=True))


@pytest.mark.parametrize("name", ["framingham", "nhanes"])
def test_split_has_no_shared_participants(name):
    d = ROOT / "data" / "processed" / name
    if not (d / "test.csv").exists():
        pytest.skip("run the pipeline first")
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    assert not set(tr["pid"]) & set(te["pid"])


def test_cluster_bootstrap_keeps_people_together():
    groups = np.array([1, 1, 2, 3, 3, 3])
    for idx in cluster_indices(groups, 20, 0):
        picked = groups[idx]
        for g in np.unique(picked):
            assert (picked == g).sum() % (groups == g).sum() == 0


def test_perfect_calibration_stats():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.5, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    s = calibration_stats(y, p)
    assert abs(s["cal_slope"] - 1) < 0.1 and abs(s["cal_intercept"]) < 0.1


@pytest.fixture(scope="module")
def client():
    if not (ROOT / "artifacts" / "framingham" / "app.json").exists():
        pytest.skip("run the pipeline first")
    from app import app
    return app.test_client()


EXAMPLES = {
    "framingham": {"male": 1, "age": 55, "sysBP": 142, "diaBP": 88, "BPMeds": 0, "prevalentHyp": 1, "currentSmoker": 1,
                   "cigsPerDay": 15, "totChol": 245, "HDL": 42, "BMI": 28.5, "diabetes": 0},
    "nhanes": {"male": 1, "age": 58, "race": 3, "education": 4, "pir": 2.5, "smoking": 2, "cigsPerDay": 15, "BPMeds": 1,
               "prevalentHyp": 1, "diabetes": 0, "famHistory": 0, "sysBP": 142, "diaBP": 84, "totChol": 225, "HDL": 42,
               "BMI": 30.1, "waist": 104, "HbA1c": 5.9, "eGFR": 78, "uacr": 22, "crp": 3.1},
}


def test_api_predict_both_models(client):
    for model in ("framingham", "nhanes"):
        r = client.post(f"/api/predict?model={model}&mode=clinician", json=EXAMPLES[model])
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        assert 0 < body["risk"] < 1
        # Shapley values add up to the gap from the typical person (log-odds)
        lo = lambda p: np.log(p / (1 - p))
        total = sum(f["contribution"] for f in body["factors"])
        assert abs(total - (lo(body["risk"]) - lo(body["typical_risk_percent"] / 100))) < 0.05
        assert body["model_probabilities"] and body["scenarios"]
        assert body["scenarios"][-1]["risk"] < body["risk"]   # healthier targets lower the risk


def test_api_rejects_missing_required(client):
    r = client.post("/api/predict?model=framingham", json={"age": 50})
    assert r.status_code == 400 and "fill in" in r.get_json()["error"]


def test_api_optional_blank_is_ok(client):
    ex = client.get("/api/schema?model=nhanes").get_json()["example"]
    ex.update({"crp": None, "uacr": "", "famHistory": None})
    r = client.post("/api/predict?model=nhanes", json=ex)
    assert r.status_code == 200 and "C-reactive protein (hs-CRP)" in r.get_json()["unknown_fields"]


def test_api_batch(client):
    ex = client.get("/api/schema?model=framingham").get_json()["example"]
    r = client.post("/api/batch?model=framingham", json={"records": [ex, {"age": 40}]}).get_json()
    assert r["results"][0]["error"] is None and r["results"][1]["error"]
