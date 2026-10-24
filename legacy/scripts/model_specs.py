"""Every model in CardioSense, defined in ONE place.

To change a model, edit it here. train.py reads this file, tunes each model
with 5-fold cross-validation on the training set, and saves the result.

Each entry has
    estimator   the scikit-learn model with its fixed settings
    grid        the hyperparameters cross-validation is allowed to choose from
    complexity  a function that ranks a setting from simple to complex
                (used by the one-standard-error rule in train.py, see below)

Overfitting controls used here
------------------------------
The goal is a model that does about as well on new patients as on the ones it
learned from. These are the levers, model by model:

    Logistic regression   C        smaller C = stronger L2 penalty on the weights
    Random forest         max_depth, min_samples_leaf, max_features
                                   shallower trees, bigger leaves, fewer features
                                   per split = less memorising
    Gradient boosting     max_depth, max_leaf_nodes, min_samples_leaf,
                          l2_regularization, learning_rate, early stopping
    Neural network        hidden_layer_sizes, alpha (L2), max_iter
                                   fewer neurons, bigger penalty, fewer epochs

Every grid deliberately reaches well past the old settings in the "simpler"
direction, so cross-validation is free to choose a much more restricted model.
"""
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import RANDOM_STATE

SPECS = {
    # ------------------------------------------------------------------
    # 1. Logistic regression: a weighted sum turned into a probability.
    # ------------------------------------------------------------------
    "logreg": dict(
        estimator=LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE),
        grid={"C": [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]},
        complexity=lambda p: (p["C"],),                    # larger C = weaker penalty
    ),

    # ------------------------------------------------------------------
    # 2. Random forest: many randomised decision trees, averaged.
    # ------------------------------------------------------------------
    "rf": dict(
        estimator=RandomForestClassifier(n_estimators=400,
                                         class_weight="balanced_subsample",
                                         n_jobs=1, random_state=RANDOM_STATE),
        grid={"max_depth": [3, 5, 8, None],
              "min_samples_leaf": [20, 40, 80],
              "max_features": ["sqrt", 0.4]},
        complexity=lambda p: (p["max_depth"] or 20, -p["min_samples_leaf"],
                              0.0 if p["max_features"] == "sqrt" else 1.0),
    ),

    # ------------------------------------------------------------------
    # 3. Gradient boosting: small trees added one at a time, each fixing the
    #    errors of the ones before. Early stopping halts it when a held-out
    #    10% of the training data stops improving.
    # ------------------------------------------------------------------
    "hgb": dict(
        estimator=HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                                 n_iter_no_change=15,
                                                 random_state=RANDOM_STATE),
        grid={"learning_rate": [0.03, 0.06],
              "max_depth": [2, 3, 5],
              "max_leaf_nodes": [7, 15],
              "min_samples_leaf": [20, 50, 100],
              "l2_regularization": [0.0, 5.0]},
        complexity=lambda p: (p["max_depth"], p["max_leaf_nodes"],
                              -p["min_samples_leaf"], -p["l2_regularization"]),
    ),

    # ------------------------------------------------------------------
    # 4. Neural network (multilayer perceptron). Early stopping on a held-out
    #    split is OFF because scikit-learn judges it by accuracy, which is
    #    nearly useless when 85% of patients are negative. The number of
    #    epochs (max_iter) is tuned by cross-validation instead.
    # ------------------------------------------------------------------
    "mlp": dict(
        estimator=MLPClassifier(early_stopping=False, solver="adam",
                                learning_rate_init=0.001, random_state=RANDOM_STATE),
        grid={"hidden_layer_sizes": [(8,), (16,), (32, 16)],
              "alpha": [0.01, 0.1, 1.0],
              "max_iter": [30, 100]},
        complexity=lambda p: (sum(p["hidden_layer_sizes"]), -p["alpha"], p["max_iter"]),
    ),
}
