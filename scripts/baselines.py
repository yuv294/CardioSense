"""Published clinical risk equations: the baselines the models must beat.

Framingham general CVD risk (D'Agostino RB et al., Circulation 2008;117:743-53)
    office  : age, BMI, systolic BP (by treatment), smoking, diabetes
    lab     : age, total cholesterol, HDL, systolic BP (by treatment), smoking, diabetes
    risk = 1 - S0 ** exp(sum(beta * x) - mean)

Pooled Cohort Equations, ACC/AHA (Goff DC et al., Circulation 2014;129:S49-73)
    race- and sex-specific (white / African American; other groups use the white
    equations, as the guideline instructs), 10-year hard ASCVD risk.

All are fixed formulas: nothing is fitted, so they can be computed for any row
without leakage. Inputs use the column names shared by both cohorts.
"""
import numpy as np
import pandas as pd

OFFICE = {
    "men":   dict(age=3.11296, bmi=0.79277, sbp_u=1.85508, sbp_t=1.92672, smoker=0.70953, diabetes=0.53160,
                  s0=0.88431, mean=23.9388),
    "women": dict(age=2.72107, bmi=0.51125, sbp_u=2.81291, sbp_t=2.88267, smoker=0.61868, diabetes=0.77763,
                  s0=0.94833, mean=26.0145),
}
LAB = {
    "men":   dict(age=3.06117, tc=1.12370, hdl=-0.93263, sbp_u=1.93303, sbp_t=1.99881, smoker=0.65451,
                  diabetes=0.57367, s0=0.88936, mean=23.9802),
    "women": dict(age=2.32888, tc=1.20904, hdl=-0.70833, sbp_u=2.76157, sbp_t=2.82263, smoker=0.52873,
                  diabetes=0.69154, s0=0.95012, mean=26.1931),
}
# Pooled Cohort Equations. Keys: ln age, ln age^2, ln TC, ln age x ln TC, ln HDL, ln age x ln HDL,
# ln SBP treated, ln age x ln SBP treated, ln SBP untreated, ln age x ln SBP untreated,
# smoker, ln age x smoker, diabetes; baseline survival; mean linear predictor.
PCE = {
    ("F", "white"): dict(a=-29.799, a2=4.884, tc=13.540, a_tc=-3.114, hdl=-13.578, a_hdl=3.149, st=2.019, a_st=0,
                         su=1.957, a_su=0, sm=7.574, a_sm=-1.665, dm=0.661, s0=0.9665, mean=-29.18),
    ("F", "black"): dict(a=17.114, a2=0, tc=0.940, a_tc=0, hdl=-18.920, a_hdl=4.475, st=29.291, a_st=-6.432,
                         su=27.820, a_su=-6.087, sm=0.691, a_sm=0, dm=0.874, s0=0.9533, mean=86.61),
    ("M", "white"): dict(a=12.344, a2=0, tc=11.853, a_tc=-2.664, hdl=-7.990, a_hdl=1.769, st=1.797, a_st=0,
                         su=1.764, a_su=0, sm=7.837, a_sm=-1.795, dm=0.658, s0=0.9144, mean=61.18),
    ("M", "black"): dict(a=2.469, a2=0, tc=0.302, a_tc=0, hdl=-0.307, a_hdl=0, st=1.916, a_st=0,
                         su=1.809, a_su=0, sm=0.549, a_sm=0, dm=0.645, s0=0.8954, mean=19.54),
}

# Typical values used only when an input a published formula needs is blank
# (fixed constants close to the cohort medians; no fitting, so no leakage).
FILL = {"age": 55.0, "sysBP": 130.0, "BMI": 27.0, "totChol": 215.0, "HDL": 50.0}


def _col(d, name, fill=np.nan):
    if name not in d:
        return np.full(len(d), fill, dtype=float)
    return pd.to_numeric(d[name], errors="coerce").fillna(fill).to_numpy(dtype=float)


def _frs(d: pd.DataFrame, lab: bool) -> np.ndarray:
    male = _col(d, "male", 0).astype(bool)
    treated = _col(d, "BPMeds", 0).astype(bool)
    smoker, diab = _col(d, "currentSmoker", 0), _col(d, "diabetes", 0)
    age, sbp = np.log(_col(d, "age")), np.log(_col(d, "sysBP"))
    out = np.full(len(d), np.nan)
    for is_male, key in ((True, "men"), (False, "women")):
        c = (LAB if lab else OFFICE)[key]
        m = male == is_male
        lp = c["age"] * age[m] + np.where(treated[m], c["sbp_t"], c["sbp_u"]) * sbp[m] \
            + c["smoker"] * smoker[m] + c["diabetes"] * diab[m]
        if lab:
            lp = lp + c["tc"] * np.log(_col(d, "totChol")[m]) + c["hdl"] * np.log(_col(d, "HDL")[m])
        else:
            lp = lp + c["bmi"] * np.log(_col(d, "BMI")[m])
        out[m] = 1.0 - c["s0"] ** np.exp(lp - c["mean"])
    return np.clip(out, 1e-4, 1 - 1e-4)


def frs_office(d):
    return _frs(d, lab=False)


def frs_lab(d):
    """NaN where HDL or total cholesterol is unknown."""
    return _frs(d, lab=True)


def frs_best(d):
    """Laboratory score when HDL is known, office score otherwise (what a clinician would use)."""
    lab = frs_lab(d)
    return np.where(np.isnan(lab), frs_office(d), lab)


def pce(d):
    male = _col(d, "male", 0).astype(bool)
    black = _col(d, "race_black", 0).astype(bool)
    treated = _col(d, "BPMeds", 0).astype(bool)
    la, ltc, lhdl, lsbp = (np.log(_col(d, c)) for c in ("age", "totChol", "HDL", "sysBP"))
    sm, dm = _col(d, "currentSmoker", 0), _col(d, "diabetes", 0)
    out = np.full(len(d), np.nan)
    for sex in ("M", "F"):
        for race in ("white", "black"):
            m = (male == (sex == "M")) & (black == (race == "black"))
            c = PCE[(sex, race)]
            lp = (c["a"] * la + c["a2"] * la ** 2 + c["tc"] * ltc + c["a_tc"] * la * ltc + c["hdl"] * lhdl
                  + c["a_hdl"] * la * lhdl + c["sm"] * sm + c["a_sm"] * la * sm + c["dm"] * dm
                  + np.where(treated, c["st"] * lsbp + c["a_st"] * la * lsbp, c["su"] * lsbp + c["a_su"] * la * lsbp))
            out[m] = 1.0 - c["s0"] ** np.exp(lp[m] - c["mean"])
    return np.clip(out, 1e-4, 1 - 1e-4)


def filled(d: pd.DataFrame) -> pd.DataFrame:
    return d.fillna({k: v for k, v in FILL.items() if k in d})
