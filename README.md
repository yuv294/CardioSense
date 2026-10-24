# CardioSense v2

Ten-year cardiovascular risk from routine health information: four machine-learning
models trained on two cohorts, benchmarked against the **Framingham Risk Score** and
the **ACC/AHA Pooled Cohort Equations** on people the models never saw, and served
through a patient portal and a clinician portal.

> Educational project. Trained on research cohorts, not clinically validated, and
> must not be used to diagnose or treat anyone.

## What changed from v1, and why

v1 trained four models on the Kaggle `framingham.csv` and none beat the Framingham
score (AUROC 0.70 vs 0.70). That was the correct result, and the literature
explains it:

* On the classic risk factors a well-built equation sits close to the information
  ceiling. Systematic reviews find no performance benefit of ML over logistic
  regression for clinical prediction (Christodoulou et al., 2019). Where ML does
  win (Weng et al., 2017; Alaa et al., 2019), the gain comes from far more data
  and far more variables.
* Framingham data is the Framingham score's **home ground**: it was derived there.

So v2 keeps the four model families but asks a better question: **does ML help when
it gets information the published scores leave out, in a population they were not
built for?** Concretely:

| | v1 | v2 |
|---|---|---|
| Data | Kaggle one-visit extract, 4,240 people | Framingham teaching dataset (3 exams, 24-yr follow-up) **and** NHANES 1999–2010 with CDC linked mortality |
| Outcome | Pre-computed `TenYearCHD` | Rebuilt from event times; prior CHD/stroke excluded; competing deaths handled |
| Events | 644 | 1,200 (Framingham) + 387 (NHANES) |
| Baselines | Framingham office score | Framingham office + lab, Pooled Cohort Equations; raw, recalibrated **and** refitted |
| Inputs | 15 | Framingham 12 (chosen by ablation); NHANES 21, incl. eGFR, urine albumin, HbA1c, hs-CRP, waist, income |
| Models | LR, RF, HGB, MLP, stack | Elastic-net LR + splines, RF (large leaves), LightGBM with monotone constraints, bagged MLP, mean ensemble |
| Validation | Random split, row bootstrap | Participant-grouped split and CV, cluster bootstrap, nested CV, calibration slope, decision curves |
| App | One calculator | Two calculators, exact Shapley explanations, what-if simulator, batch CSV scoring, evidence dashboard, Docker |

The v1 code, results and documents are kept in [`legacy/`](legacy/).

## Results (locked test sets, opened once)

**NHANES (external cohort), 10-year CVD death.** 2,740 people, 77 deaths.

| Model | AUROC (95% CI) | Brier | E/O |
|---|---|---|---|
| Logistic regression | 0.851 (0.803–0.892) | 0.0246 | 1.00 |
| **Ensemble (served)** | **0.848 (0.797–0.891)** | 0.0250 | 1.00 |
| Gradient boosting | 0.846 (0.799–0.888) | 0.0252 | 1.04 |
| Random forest | 0.840 (0.792–0.884) | 0.0254 | 1.03 |
| PCE, refitted | 0.833 (0.787–0.873) | 0.0256 | 1.02 |
| Pooled Cohort Equations (published) | 0.828 (0.781–0.869) | 0.0382 | 3.85 |
| Neural network | 0.812 (0.760–0.859) | 0.0257 | 0.94 |
| Framingham (published) | 0.800 (0.751–0.847) | 0.0524 | 5.47 |

* Served ensemble vs published PCE: **ΔAUROC +0.020 (95% CI +0.002 to +0.038, p = 0.03)**; Brier −0.013.
* vs PCE recalibrated to the cohort: same ΔAUROC; Brier −0.0008 (CI −0.0014 to −0.0002).
* vs the PCE formula *refitted* on our data: +0.015 (CI −0.006 to +0.037) on the test set;
  nested CV (below) confirms it, +0.031 (CI +0.020 to +0.043). The gain comes from the new
  inputs, not from refitting.
* The published scores over-predict CVD *death* 4–5× because they predict fatal
  **and** non-fatal events; recalibration fixes the scale but cannot change ranking.

**Framingham (home ground), 10-year first CHD event.** 839 people (2,128 visits), 245 events.

| Model | AUROC (95% CI) | Brier | E/O |
|---|---|---|---|
| **Ensemble (served)** | **0.722 (0.685–0.758)** | 0.0954 | 1.00 |
| Framingham formula, refitted | 0.722 (0.683–0.759) | 0.0957 | 1.02 |
| Logistic regression | 0.720 | 0.0956 | 1.02 |
| Gradient boosting | 0.720 | 0.0957 | 1.03 |
| Random forest | 0.717 | 0.0959 | 1.03 |
| Framingham score (published) | 0.709 (0.672–0.744) | 0.1050 | 1.52 |
| Neural network | 0.705 | 0.0963 | 0.93 |

* On this test split the served ensemble beats published Framingham by ΔAUROC +0.013
  (CI +0.001 to +0.025, p = 0.03). **The nested CV, which uses all 1,200 events, shrinks
  that to +0.003 (CI −0.003 to +0.009): on Framingham data discrimination is a tie.** The
  test-split gain was partly the luck of the split, which is exactly why both are reported.
* The real, robust gain on Framingham is **calibration**: the published score over-predicts
  first CHD events by about 50% (E/O 1.48–1.52), while every model is at 1.00 (Brier −0.007
  in nested CV, CI excludes 0). A calibrated percentage is what a patient actually sees.
* The ensemble ties the **refitted** Framingham formula. The ablation shows why:
  restricting the model to the published score's own inputs costs nothing, because in
  this cohort those inputs are all the information there is.

**Where the signal comes from** (`ablation.py`, training participants, 3 × 5-fold grouped CV):

| Change | NHANES, boosting | NHANES, logistic | Framingham, boosting |
|---|---|---|---|
| Only the published score's inputs | **−0.031** | **−0.021** | +0.001 |
| Without kidney health (eGFR, urine albumin) | −0.005 | −0.010 | n/a |
| Without inflammation (hs-CRP) | −0.007 | −0.001 | n/a |
| Without social factors (education, income) | −0.004 | −0.004 | n/a |
| Without blood pressure | −0.009 | −0.002 | −0.014 |

The two cohorts mirror each other: in Framingham the published inputs are all the
information there is; in NHANES the inputs the published scores leave out are worth
0.02–0.03 AUROC. That matches the direction of the 2023 AHA PREVENT equations, which
added kidney function and HbA1c.

**Nested cross-validation** (the whole tuning procedure repeated in 5 outer folds, so
every participant's prediction is out-of-sample and every event counts) is in
`reports/<cohort>/nested_cv.json` and on the Evidence page.

| Nested CV, all events | Served model | Published score | ΔAUROC (95% CI) | vs refitted formula |
|---|---|---|---|---|
| NHANES, 387 deaths | 0.817 | PCE 0.793 | **+0.024 (+0.013 to +0.035), p < 0.001** | **+0.031 (+0.020 to +0.043)** |
| Framingham, 1,200 events | 0.720 | FRS 0.717 | +0.003 (−0.003 to +0.009), n.s. | −0.002 (−0.006 to +0.002) |

The nested CV settles both open questions. On NHANES the gain over the PCE is real and
it is **not** a refitting effect: the refitted PCE formula does no better than the
published one (−0.007), while the models with the extra inputs gain +0.03. On Framingham
discrimination is a tie, and the gain is in calibration only. Across both cohorts the
neural network is the weakest model (Grinsztajn et al., 2022), and logistic regression is
as good as boosting (Christodoulou et al., 2019): information mattered more than the algorithm.

**Overfitting check.** Train / CV / test AUROC gaps are small for every model (e.g.
NHANES ensemble 0.85 / 0.80 / 0.85; Framingham 0.74 / 0.72 / 0.72. NHANES test AUROCs sit slightly above CV for every model, including the refitted PCE, so that test split is simply a somewhat easier sample, not a sign of leakage). Settings are
chosen with the one-standard-error rule, which always lands on the more regularised
end of each grid.

## How it works

```
scripts/
  cohorts/framingham.py   cohort 1: build table, inputs, form metadata, features, baselines
  cohorts/nhanes.py       cohort 2: harmonise 6 NHANES cycles, eGFR (CKD-EPI 2021), mortality link
  nhanes_download.py      fetch the public NHANES + NCHS mortality files (~160 MB)
  baselines.py            Framingham office/lab (D'Agostino 2008), PCE (Goff 2014)
  models.py               the four model families + refitted published formula
  calibration.py          cross-validated Platt recalibration, mean ensemble
  prepare.py              build + participant-level 80/20 split
  train.py                parallel grouped-CV tuning, one-SE rule, calibration, served-model choice
  evaluate.py             locked test set: AUROC/Brier/calibration/DCA/subgroups, cluster bootstrap
  nested_cv.py            nested CV over the whole cohort
  ablation.py             which inputs carry the signal
  export.py, figures.py   app artifacts and PNG figures
  run_pipeline.py         everything, in order
backend/                  Flask API (also serves the built frontend)
frontend/                 React + Vite + Tailwind + Recharts
tests/                    pytest: equations vs published examples, leakage checks, API
```

Key design choices (each is explained in the code and on the app's Method page):

* **Landmarking** (van Houwelingen 2007): each Framingham visit is a starting point, so
  people contribute up to three rows; splits and CV folds keep a person's rows together.
* **Proper scoring**: hyperparameters are chosen on log-loss, not accuracy or AUROC alone.
* **Monotone constraints** in gradient boosting: risk can only rise with age, blood
  pressure, smoking, HbA1c, urine albumin…, which keeps what-if answers sane.
* **Missing optional inputs stay missing**: the models are trained on rows where they
  are genuinely missing, so a blank lab result is a supported input, not a guess.
* **Exact group Shapley explanations**: contributions of plain-language factors (age,
  blood pressure, kidney health…) against the cohort's typical person; they add up
  exactly to the person's risk gap, for any model.

## Run it

Requires Python 3.10+ and Node 18+.

```bash
python -m venv venv && venv/Scripts/activate        # Windows; use venv/bin/activate on macOS/Linux
pip install -r backend/requirements.txt
cd scripts && python run_pipeline.py                 # downloads NHANES, trains, evaluates (~1 h; --skip-nested ~25 min)
cd ../frontend && npm install && npm run build
cd ../backend && python app.py                       # http://127.0.0.1:5000 serves the API and the app
```

For frontend development run `npm run dev` in `frontend/` (http://localhost:5173; it
proxies `/api` to the backend). Tests: `pytest -q` from the repository root.

**Docker** (after the pipeline has produced `artifacts/` and `reports/`):

```bash
docker build -t cardiosense . && docker run -p 8000:8000 cardiosense
```

`render.yaml` deploys the same image on Render's free tier.

## Data sources

* Framingham Heart Study teaching dataset (NHLBI), distributed in the CRAN package
  `riskCommunicator`. Teaching data: anonymised and not for publication.
* NHANES 1999–2010 (CDC/NCHS) and the NCHS public-use linked mortality files (follow-up to 31 Dec 2019).

## Limitations

Both cohorts are American and Framingham is mostly white and historical. NHANES deaths
come from death certificates (non-fatal events are not counted). Family history was asked
only from 2005. Survey weights are not used. What-if figures are associations, not trial
effects. The NHANES test set holds 77 deaths, so its intervals are wide; nested CV is the
more precise estimate.

## Literature

D'Agostino 2008 (Circulation) · Goff 2014 (Circulation) · Khan 2024, PREVENT (Circulation) ·
Christodoulou 2019 (J Clin Epidemiol) · Weng 2017 (PLoS One) · Alaa 2019 (PLoS One) ·
Grinsztajn 2022 (NeurIPS) · van Houwelingen 2007 (Scand J Stat) · Van Calster 2019 (BMC Med) ·
Vickers & Elkin 2006 (Med Decis Making) · Riley 2020 (BMJ) · Collins 2024, TRIPOD+AI (BMJ).

## Team

* Rosanne Chugh: frontend, backend, integration
* Rajapurohitham Sahana and Yuvika Das: machine-learning models and evaluation
