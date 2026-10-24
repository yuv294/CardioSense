"""Every model in CardioSense v2, defined in one place.

The four model families from v1 are kept (logistic regression, random forest,
gradient boosting, neural network) but each is rebuilt around what the
literature says works on small tabular clinical data:

* logreg  - elastic-net logistic regression on the Framingham functional form
            (logs, sex interactions) plus spline terms for age, BP, cholesterol
            and BMI. Penalisation, not feature dropping, controls overfitting.
* rf      - random forest with large leaves (min_samples_leaf >= 20). Fully
            grown trees memorise a 13%-event dataset; big leaves give smooth
            probabilities.
* gbm     - LightGBM with monotone constraints on clinically certain inputs,
            shallow trees, strong regularisation and native missing-value
            handling. (Grinsztajn et al. 2022: boosted trees remain the
            strongest learners on tabular data.)
* mlp     - a small neural network with weight decay and early stopping,
            bagged over 5 seeds to cut variance.
* refit   - not ML: the published score's own variables in its own form
            (Framingham / PCE), with coefficients re-estimated on our training
            data. It answers "is the gain from machine learning and new inputs,
            or just from refitting the old formula?"

Each entry: a factory make(cohort, **setting) returning an unfitted estimator,
and a small search space. Cohort modules supply the columns, monotone
directions, spline columns and refit columns (see cohorts/__init__.py).
Hyperparameters are chosen by grouped cross-validation on log-loss (a proper
scoring rule, which rewards calibrated probabilities, not just ranking).
"""
import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer

from config import RANDOM_STATE


def feature_columns(c):
    """Model-matrix columns of a cohort (computed from one dummy row)."""
    return list(c.add_features(pd.DataFrame([{f: 1.0 for f in c.FEATURES}])).columns)


def _impute_scale():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())


def make_logreg(c, C=0.1, l1_ratio=0.5):
    pre = ColumnTransformer([
        ("all", _impute_scale(), feature_columns(c)),
        ("spl", make_pipeline(SimpleImputer(strategy="median"),
                              SplineTransformer(n_knots=4, degree=3, include_bias=False), StandardScaler()),
         c.SPLINE_COLS),
    ])
    lr = LogisticRegression(solver="saga", l1_ratio=l1_ratio, C=C, max_iter=5000)
    return Pipeline([("pre", pre), ("clf", lr)])


def make_rf(c, min_samples_leaf=40, max_features=0.33):
    pre = SimpleImputer(strategy="median", add_indicator=True)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                n_jobs=-1, random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", rf)])


def make_gbm(c, num_leaves=7, learning_rate=0.02, n_estimators=600, min_child_samples=60, reg_lambda=5.0,
             colsample_bytree=0.6):
    mono = [c.MONOTONE.get(col, 0) for col in feature_columns(c)]
    return LGBMClassifier(num_leaves=num_leaves, max_depth=3, learning_rate=learning_rate, n_estimators=n_estimators,
                          min_child_samples=min_child_samples, reg_lambda=reg_lambda, subsample=0.8, subsample_freq=1,
                          colsample_bytree=colsample_bytree, monotone_constraints=mono,
                          monotone_constraints_method="advanced", random_state=RANDOM_STATE, n_jobs=4, verbose=-1)


def make_mlp(c, alpha=1.0, hidden=(32,)):
    net = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=20, max_iter=500, learning_rate_init=1e-3, random_state=RANDOM_STATE)
    bag = BaggingClassifier(net, n_estimators=5, max_samples=1.0, bootstrap=False, random_state=RANDOM_STATE)
    return Pipeline([("pre", _impute_scale()), ("clf", bag)])


def make_refit(c, C=1.0):
    pre = ColumnTransformer([("frs", _impute_scale(), c.REFIT_COLS)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=2000))])


SPECS = {
    "logreg": {"make": make_logreg, "space": {"C": [0.003, 0.01, 0.03, 0.1, 0.3], "l1_ratio": [0.0, 0.5]}},
    "rf": {"make": make_rf, "space": {"min_samples_leaf": [20, 40, 80, 160], "max_features": [0.2, 0.33, 0.5]}},
    "gbm": {"make": make_gbm, "space": {"num_leaves": [3, 5, 7], "learning_rate": [0.01, 0.02, 0.04],
                                        "n_estimators": [300, 600, 1000], "min_child_samples": [40, 80, 160],
                                        "reg_lambda": [1.0, 5.0, 20.0], "colsample_bytree": [0.4, 0.6, 0.8]}},
    "mlp": {"make": make_mlp, "space": {"alpha": [0.1, 1.0, 3.0, 10.0], "hidden": [(16,), (32,), (32, 16)]}},
    "refit": {"make": make_refit, "space": {"C": [1.0]}},
}

LABELS = {"logreg": "Logistic regression (elastic net + splines)", "rf": "Random forest",
          "gbm": "Gradient boosting (LightGBM, monotone)", "mlp": "Neural network (bagged MLP)",
          "ensemble": "Ensemble (mean of the four)", "refit": "Published formula, refitted",
          "frs_best": "Framingham score (published)", "frs_office": "Framingham office score (published)",
          "frs_lab": "Framingham lab score (published)", "pce": "Pooled Cohort Equations (published)"}

BASE_MODELS = ["logreg", "rf", "gbm", "mlp"]


def single_thread(est):
    """For parallel tuning: one thread per fit (the parallelism is across fits)."""
    for k in est.get_params():
        if k.endswith("n_jobs"):
            est.set_params(**{k: 1})
    return est


def sample_space(space, n, rng):
    """Random search: n distinct settings (or the full grid if smaller)."""
    keys = list(space)
    grid = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    if len(grid) <= n:
        return grid
    return [grid[i] for i in rng.choice(len(grid), size=n, replace=False)]
