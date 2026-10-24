"""Cross-validated Platt recalibration, as a picklable wrapper.

The base model is refitted on all training rows; the logistic map
(intercept a, slope b on the log-odds scale) is fitted on that model's
OUT-OF-FOLD predictions, so the calibration never sees a row the model was
trained on. This is the same idea as sklearn's CalibratedClassifierCV but keeps
one refitted model (faster to serve, and easier to explain).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_platt(p_oof, y):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_oof).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def apply_platt(p, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * logit(p))))


class Calibrated:
    def __init__(self, base, a, b):
        self.base, self.a, self.b = base, a, b
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = apply_platt(self.base.predict_proba(X)[:, 1], self.a, self.b)
        return np.column_stack([1 - p, p])


class MeanEnsemble:
    """Unweighted mean of calibrated members: no fitted weights, nothing extra to overfit."""

    def __init__(self, members):
        self.members = members
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members.values()], axis=0)
        return np.column_stack([1 - p, p])
