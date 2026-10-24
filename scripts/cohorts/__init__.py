"""Cohort definitions. Each module exposes the same interface:

    NAME, TITLE, OUTCOME, SOURCE         descriptive text for reports and the app
    FEATURES, OPTIONAL, FEATURE_META     the raw inputs and how the form asks for them
    FORM_GROUPS                          [(group key, title, note)] for the form layout
    MODIFIABLE                           inputs the what-if simulator may change
    FACTOR_OF                            raw input -> plain-language factor (explanations)
    add_features(df) -> DataFrame        row-wise feature engineering (train AND serve)
    MONOTONE, REFIT_COLS, SPLINE_COLS    model hints (see models.py)
    BASELINES                            {name: fn(df) -> 10-year risk} published equations
    PRIMARY_BASELINE                     the one the headline comparison uses
    STRATA                               extra columns kept for subgroup analysis
    build() -> (df, log)                 analysis table: pid, y, FEATURES, STRATA
"""
import importlib


def get(name: str):
    return importlib.import_module(f"cohorts.{name}")
