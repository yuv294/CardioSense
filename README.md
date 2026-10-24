# CardioSense

Ten-year coronary heart disease (CHD) risk estimation from routine health
information, with separate portals for patients and clinicians.

The system estimates the probability that a person develops CHD within ten years
using only screening-level inputs: age, sex, blood pressure, cholesterol,
smoking, diabetes, BMI and a few others. No stress test or angiogram is needed.
Five machine-learning models are compared against a published clinical score
(the Framingham Risk Score). The best-calibrated model is served through a Flask
API and a React web app.

This is an educational project. It is trained on a historical teaching cohort,
has not been clinically validated, and must not be used to diagnose or treat
anyone.

## Why the evaluation can be trusted

Many "heart disease" student projects predict disease that is already present,
from test results that contain the answer, and report 90%+ accuracy. This
project predicts a future event from everyday inputs, which is harder, and holds
itself to a real benchmark.

- A locked test set (20%, stratified) is opened once, at evaluation. No tuning
  decision sees it.
- Imputation and scaling are fit on training data only, so nothing leaks.
- The Framingham Risk Score is scored on the same test set, so "is the ML worth
  it?" is answered rather than assumed.
- Calibration is measured, because the app shows a real percentage to a person.

## Results (locked test set, n = 848, 15.2% developed CHD)

| Model | AUROC | 95% CI | AUPRC |
|---|---|---|---|
| Framingham Risk Score (baseline) | 0.703 | 0.66 - 0.75 | 0.299 |
| Stacked ensemble | 0.696 | 0.64 - 0.74 | 0.299 |
| Logistic regression (served, calibrated) | 0.695 | 0.65 - 0.74 | 0.292 |
| Neural network (MLP) | 0.693 | 0.64 - 0.74 | 0.299 |
| Gradient boosting | 0.689 | 0.64 - 0.74 | 0.288 |
| Random forest | 0.680 | 0.63 - 0.73 | 0.297 |

No machine-learning model clearly beats the Framingham score on the test set;
the confidence intervals overlap almost entirely (paired difference between the
served model and Framingham: -0.006, 95% interval -0.024 to +0.013). On standard
risk factors a well-built clinical rule is a strong baseline, which matches the
published literature.

At the screening threshold of 15.5% the served model flags 61% of future cases
(sensitivity) and clears 66% of people who stay healthy (specificity).
Isotonic calibration cut the served model's Brier score from 0.212 to 0.123.

### Can it be improved? The improvement ladder

The test set is too small (129 events) to see improvements of 0.005 to 0.01
AUROC, so ideas are judged on the training rows only, with 5 x 5 repeated
cross-validation, nested tuning and a paired bootstrap against Framingham
(`scripts/ladder.py`).

| Idea | AUROC (development) | Gain over previous model |
|---|---|---|
| Framingham, as published | 0.721 | |
| Original 23-feature logistic regression | 0.729 | |
| + Framingham-structure features (served) | 0.734 | +0.004 (95% interval +0.001 to +0.009) |
| + missing-value flags | 0.734 | none |
| + Framingham score as a feature | 0.734 | none |
| Framingham score as a fixed offset | 0.727 | worse |
| Boosting (with or without the score) | 0.723 to 0.724 | none |

Only copying the structure of the Framingham equation (logs, separate effects
for men and women, treated blood pressure) helped. The served model is ahead of
Framingham in development cross-validation (+0.013, interval +0.003 to +0.023)
but not on the test split, so the honest summary is a tie to a small edge. The
selection rule was amended after seeing the results; see `MODELS_EXPLAINED.md`
Part 15. `scripts/ceiling.py` shows the limit is the inputs, not the model: five
inputs already reach 0.732 and the learning curve is flat. `scripts/subgroups.py`
shows that the published score under-predicts risk in this cohort and that
calibration is weaker in women and people aged 60 and over.

### Overfitting is controlled in the code

Every model is defined in `scripts/model_specs.py` with a search grid that
includes very simple settings, and `train.py` picks the simplest setting within
one standard error of the best cross-validated score. `scripts/diagnostics.py`
prints train, cross-validated and test AUROC for each model. The train-to-test
gap of the random forest fell from 0.21 to 0.07 and of gradient boosting from
0.17 to 0.07 (random forest 0.067, gradient boosting 0.065). To change or improve a model, edit its entry in `model_specs.py`
and re-run the scripts (see the table below).

## Project layout

```
data/raw/framingham.csv        teaching cohort (4,240 patients, 15 predictors)
scripts/
  config.py                    paths, feature list, random seed
  features.py                  engineered and Framingham-structure features, shared by training and serving
  preprocess.py                split first, fit on train only
  baseline_frs.py              Framingham Risk Score (office/BMI model)
  model_specs.py               definition and search grid of every model
  train.py                     5-fold CV tuning, one-standard-error rule, stacking
  diagnostics.py               train / CV / test AUROC per model (overfitting check)
  evaluate.py                  locked-test metrics, calibration, threshold
  explain.py                   per-feature contributions for the app
  population.py                cohort percentiles and subgroup event rates
  make_plots.py                report figures
  offset_logreg.py             logistic regression with a fixed-weight offset
  ladder.py                    improvement ladder (repeated nested CV, paired bootstrap)
  subgroups.py                 calibration and net benefit by sex and age band
  ceiling.py                   input ablation and learning curve
  paired_test.py               paired test-set comparison with Framingham
artifacts/                     fitted preprocessor, models, explainer
reports/                       evaluation.json, metrics table, figures
backend/                       Flask API
frontend/                      React + Vite + Tailwind
```

## The app

**Patient portal**
- Assess: a short form, then the risk percentage, a meter showing where it falls,
  what drove the result, and how the person compares with the study cohort.
- History: saved results and a risk-over-time chart.
- Learn: plain-language explanations of each measurement.
- Download a one-page PDF summary.

**Clinician portal**
- Assess patient: the decision panel with the screening-threshold flag, agreement
  between models, risk percentile, every model's raw estimate, event rate in
  similar patients, and the contribution of all 31 features.
- Patient records: every assessment by patient, change since the last visit,
  CSV export.
- Model performance: model comparison, calibration, and outcomes at the threshold.

Each portal has its own sign-in. Accounts and saved records are stored in the
browser (localStorage), not on a server.

## Run it locally

Requires Python 3.10+ and Node 18+.

Train the models (one time; writes everything in `artifacts/` and `reports/`):

```
cd scripts
pip install -r ../backend/requirements.txt
python preprocess.py
python baseline_frs.py
python train.py
python evaluate.py
python explain.py
python population.py
python make_plots.py
python diagnostics.py
python ladder.py        # optional: improvement ladder (about 10 minutes)
python subgroups.py     # optional
python ceiling.py       # optional
python paired_test.py   # optional
```

Start the backend at http://127.0.0.1:5000:

```
cd backend
python app.py
```

Start the frontend at http://localhost:5173:

```
cd frontend
npm install
npm run dev
```

The dev server forwards `/api` to the backend. See `DEPLOYMENT.md` for deploying
on free tiers.

## Data

Framingham Heart Study teaching dataset: 4,240 participants, 15 predictors,
outcome `TenYearCHD`.

## Team

- Rosanne Chugh: frontend, backend, integration
- Rajapurohitham Sahana and Yuvika Das: machine-learning models and evaluation
