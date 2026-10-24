"""Step 3 of the pipeline: tune five models on the TRAINING set only.

The models and their search grids live in model_specs.py (edit them there).

How each model is chosen
------------------------
1. GridSearchCV tries every combination in the grid with 5-fold stratified
   cross-validation, scoring each by average precision (area under the
   precision-recall curve), which respects the 15% class imbalance.
2. Taking the single best score tends to pick the most flexible model that got
   lucky in the folds. To resist that we apply the ONE-STANDARD-ERROR RULE
   (Hastie, Tibshirani & Friedman, "The Elements of Statistical Learning",
   section 7.10): among all settings whose cross-validated score is within one
   standard error of the best, choose the SIMPLEST. Simpler settings overfit
   less and cost almost nothing in CV score.
3. The chosen setting is refitted on the whole training set.
4. A stacked ensemble combines the four tuned models with a logistic
   meta-learner trained on out-of-fold predictions.

The locked test set is never touched here.

Outputs (artifacts/)
    model_{logreg,rf,hgb,mlp,stack}.joblib
    cv_results.json  -- chosen settings, CV scores, train scores, and the
                        setting the plain "best score" rule would have chosen
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

from config import DATA_PROCESSED, ARTIFACTS, RANDOM_STATE
from model_specs import SPECS

warnings.filterwarnings("ignore", category=ConvergenceWarning)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = "average_precision"


def load_train():
    X = np.load(DATA_PROCESSED / "X_train_proc.npy")
    y = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0].to_numpy()
    return X, y


def tune(name, spec, X, y):
    gs = GridSearchCV(spec["estimator"], spec["grid"], scoring=SCORING, cv=CV,
                      n_jobs=-1, refit=False, return_train_score=True)
    gs.fit(X, y)
    r = gs.cv_results_
    mean, std = r["mean_test_score"], r["std_test_score"]
    best = int(np.argmax(mean))
    se = std[best] / np.sqrt(CV.get_n_splits())
    near_best = [i for i in range(len(mean)) if mean[i] >= mean[best] - se]
    pick = min(near_best, key=lambda i: spec["complexity"](r["params"][i]))

    model = clone(spec["estimator"]).set_params(**r["params"][pick]).fit(X, y)
    info = {
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in r["params"][pick].items()},
        "cv_average_precision": float(mean[pick]),
        "train_average_precision": float(r["mean_train_score"][pick]),
        "one_se": float(se),
        "settings_within_one_se": len(near_best),
        "settings_tried": len(mean),
        "best_score_rule_params": {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in r["params"][best].items()},
        "best_score_rule_cv_ap": float(mean[best]),
    }
    print(f"  {name:7s} chose {info['params']}")
    print(f"          CV AP {mean[pick]:.4f}  (best possible {mean[best]:.4f}, "
          f"1 SE = {se:.4f}); train AP {info['train_average_precision']:.4f}; "
          f"{len(near_best)} of {len(mean)} settings within 1 SE")
    return model, info


def main():
    X, y = load_train()
    print(f"Training on {len(y)} rows, {y.mean():.1%} positive\n")

    models, results = {}, {}
    for name, spec in SPECS.items():
        models[name], results[name] = tune(name, spec, X, y)

    # --- stacking ensemble --------------------------------------------------
    # The meta-learner trains on the base models' out-of-fold predictions, so a
    # base model never feeds it a prediction for rows it was trained on.
    stack = StackingClassifier(
        estimators=[(n, models[n]) for n in ("logreg", "rf", "hgb", "mlp")],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=RANDOM_STATE),
        stack_method="predict_proba", cv=CV, n_jobs=1, passthrough=False,
    )
    stack_cv = cross_val_score(stack, X, y, scoring=SCORING, cv=CV, n_jobs=1)
    stack.fit(X, y)
    models["stack"] = stack
    results["stack"] = {"params": "stacking(logreg,rf,hgb,mlp) -> logreg meta",
                        "cv_average_precision": float(stack_cv.mean())}
    print(f"  stack   CV AP {stack_cv.mean():.4f}")

    for name, model in models.items():
        joblib.dump(model, ARTIFACTS / f"model_{name}.joblib")
    (ARTIFACTS / "cv_results.json").write_text(json.dumps(results, indent=2))

    print("\nCV ranking (average precision):")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["cv_average_precision"]):
        print(f"  {name:7s} {r['cv_average_precision']:.4f}")


if __name__ == "__main__":
    main()
