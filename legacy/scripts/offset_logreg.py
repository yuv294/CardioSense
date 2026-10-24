"""Logistic regression with an OFFSET (a fixed starting score).

Idea (clinical "model updating"): do not relearn what the Framingham score
already knows. Treat its log-odds as a fixed starting point whose weight is 1,
learn only an intercept and small, penalised corrections from our other
features. With a strong penalty (small C) the model stays close to Framingham
and moves away only where the data clearly disagrees.

    z = offset + intercept + sum_j beta_j * x_j        (offset has weight exactly 1)

The LAST column of X is the offset (the Framingham log-odds, not scaled). All
other columns are the usual standardised features. The objective matches
scikit-learn's LogisticRegression so the meaning of C is the same:
    C * sum_i w_i * logloss_i + 0.5 * ||beta||^2     (intercept not penalised)
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class OffsetLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, class_weight="balanced", max_iter=500):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        off, F = X[:, -1], X[:, :-1]
        n, p = F.shape
        if sample_weight is not None:
            w = np.asarray(sample_weight, float)
        elif self.class_weight == "balanced":
            cnt = np.bincount(y, minlength=2)
            w = (n / (2.0 * cnt))[y]
        else:
            w = np.ones(n)
        C = self.C

        def obj(theta):
            b, beta = theta[0], theta[1:]
            z = off + b + F @ beta
            pr = _sigmoid(z)
            loss = -(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))
            r = w * (pr - y)
            val = C * np.sum(w * loss) + 0.5 * beta @ beta
            grad = np.concatenate(([C * r.sum()], C * (F.T @ r) + beta))
            return val, grad

        res = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:].reshape(1, -1)
        self.classes_ = np.array([0, 1])
        self.n_iter_ = np.array([res.nit])
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        return X[:, -1] + self.intercept_ + X[:, :-1] @ self.coef_[0]

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
