"""Step 1 of the pipeline: split, then fit transformers on TRAIN ONLY.

Why this order matters
----------------------
The single most common leak in student ML projects is fitting a scaler or an
imputer on the whole dataset and *then* splitting. That lets information from
the test rows (their medians, their min/max) seep into the training pipeline,
so the final test score is optimistic and the project's headline number is a
lie. We split first, lock the test set away, and fit every transformer on the
training rows alone. The test set is transformed using those train-derived
statistics -- never its own.

Outputs (to data/processed/)
    X_train.csv, X_test.csv        raw feature columns, pre-transform (kept for
                                   the rule-based baseline, which wants raw units)
    y_train.csv, y_test.csv        labels
    X_train_proc.npy, X_test_proc  imputed + scaled arrays for the ML models
    preprocessor.joblib            the fitted ColumnTransformer (imputer+scaler)
    feature_defaults.json          train medians/modes, for the live API to fill
                                   missing inputs with the SAME values used in training
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_RAW, DATA_PROCESSED, ARTIFACTS, FEATURES, ENGINEERED_FEATURES, TARGET,
    RANDOM_STATE, TEST_SIZE, FEATURE_META,
)
from features import add_engineered_features, add_structure_features, STRUCTURE_FEATURES

# The model sees the 15 raw features PLUS the engineered ones.
# 15 raw + 8 engineered + 8 Framingham-style structure features (log scales, sex interactions,
# treated blood pressure). The structure layer was added after the improvement ladder showed it
# helps (ladder.py, step S3).
ALL_FEATURES = FEATURES + ENGINEERED_FEATURES + STRUCTURE_FEATURES

# Columns that are 0/1 flags or small integer codes -- impute with the mode,
# never scale (scaling a yes/no flag is meaningless). Everything else is
# continuous: impute with the median, then standardise. The engineered flags
# (hyp_and_diab, high_glucose) are binary-like; the rest are continuous.
BINARY_LIKE = ["male", "currentSmoker", "BPMeds", "prevalentStroke",
               "prevalentHyp", "diabetes", "education",
               "hyp_and_diab", "high_glucose", "male_x_smoker", "male_x_diabetes"]
CONTINUOUS = [f for f in ALL_FEATURES if f not in BINARY_LIKE]


def main():
    df = pd.read_csv(DATA_RAW)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # engineer features on the full frame first (deterministic, no stats), then
    # select the full feature set. Engineering before the split is safe because
    # it uses no cross-row information -- each row's engineered values depend
    # only on that same row's raw values.
    df = add_structure_features(add_engineered_features(df))
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # --- split FIRST, stratified so both halves keep the ~15% CHD rate ------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows ({y_train.mean():.1%} positive)")
    print(f"Test : {X_test.shape[0]} rows ({y_test.mean():.1%} positive)  [LOCKED]")

    # --- fit transformers on TRAIN ONLY -------------------------------------
    continuous_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    pre = ColumnTransformer([
        ("cont", continuous_pipe, CONTINUOUS),
        ("bin", binary_pipe, BINARY_LIKE),
    ])

    X_train_proc = pre.fit_transform(X_train)   # fit uses train rows only
    X_test_proc = pre.transform(X_test)         # test transformed w/ train stats

    # processed column order = continuous block, then binary block
    proc_cols = CONTINUOUS + BINARY_LIKE

    # --- defaults for the live API: medians/modes from TRAIN ----------------
    # The API receives only the 15 RAW inputs and fills any blank with these,
    # THEN engineers features from the completed raw row. So defaults are needed
    # for raw columns only; engineered columns are always recomputed at serve
    # time and never filled from a stored default.
    defaults = {}
    for col in FEATURES:
        if col in BINARY_LIKE:
            defaults[col] = float(X_train[col].mode(dropna=True).iloc[0])
        else:
            defaults[col] = float(np.nanmedian(X_train[col]))

    # --- save everything ----------------------------------------------------
    X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)
    np.save(DATA_PROCESSED / "X_train_proc.npy", X_train_proc)
    np.save(DATA_PROCESSED / "X_test_proc.npy", X_test_proc)
    joblib.dump(pre, ARTIFACTS / "preprocessor.joblib")
    (ARTIFACTS / "processed_columns.json").write_text(json.dumps(proc_cols, indent=2))
    (ARTIFACTS / "feature_defaults.json").write_text(json.dumps(defaults, indent=2))

    print("\nSaved processed arrays, fitted preprocessor, and feature defaults.")
    print(f"Processed feature order: {proc_cols}")


if __name__ == "__main__":
    main()
