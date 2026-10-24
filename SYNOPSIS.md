# CardioSense v2: project synopsis

**Submitted by:** Yuvika Das · Rajapurohitham Sahana · Rosanne Chugh

## Abstract

Clinical risk equations such as the Framingham Risk Score (FRS) and the ACC/AHA Pooled
Cohort Equations (PCE) are the standard for estimating ten-year cardiovascular risk. Our
first version found, like much of the literature, that machine learning (ML) trained on
the same classic risk factors could not beat the FRS. Version 2 asks when ML *does* add
value. We train four model families (elastic-net logistic regression with splines, random
forest, monotone-constrained gradient boosting, bagged neural network) and their ensemble
on two cohorts: the Framingham Heart Study teaching dataset (the FRS's home ground; 10-year
first coronary event; 1,200 events) and NHANES 1999–2010 linked to national death records
(an external, multi-ethnic, modern cohort with kidney, glycaemic, inflammatory and social
markers; 10-year cardiovascular death; 387 deaths). On Framingham, ML ties the published
FRS in discrimination (nested-CV ΔAUROC +0.003) but is far better calibrated (the FRS
over-predicts by ~50%). On NHANES, the ensemble beats the published PCE on a locked test
set (AUROC 0.848 vs 0.828, Δ +0.020, 95% CI +0.002 to +0.038) and in nested cross-validation over all 387 deaths (Δ +0.024, 95% CI +0.013 to +0.035); an
ablation attributes the gain to inputs the equations omit (urine albumin, eGFR, hs-CRP,
waist, income and education). The models are served through a Flask API and a React web
app with exact Shapley explanations, a what-if simulator and batch scoring.

## 1. Problem

Can machine learning improve ten-year cardiovascular risk estimation over the published
equations clinicians use, and if so, where does the improvement come from: the algorithm,
refitting to the population, or new information?

## 2. Literature

* Published equations: D'Agostino 2008 (FRS), Goff 2014 (PCE), Khan 2024 (AHA PREVENT, which
  adds eGFR, urine albumin and HbA1c).
* ML vs regression: Christodoulou 2019 (systematic review: no average benefit of ML over
  logistic regression); Weng 2017 and Alaa 2019 (ML gains with very large data and many
  variables); Grinsztajn 2022 (tree ensembles lead on tabular data).
* Methods: van Houwelingen 2007 (landmarking), Riley 2020 (sample size), Van Calster 2019
  (calibration), Vickers & Elkin 2006 (decision curves), Collins 2024 (TRIPOD+AI reporting).

## 3. Data

| | Framingham teaching dataset | NHANES 1999–2010 + NCHS linked mortality |
|---|---|---|
| People / rows | 4,199 / 10,642 visits (landmarking) | 13,702 |
| Outcome | First CHD event ≤ 10 years | CVD death ≤ 10 years |
| Events | 1,200 (11.3%) | 387 (2.8%) |
| Inputs | 12 (chosen by ablation) | 21, 8 optional |
| Exclusions | Prior CHD or stroke | Prior CVD; age outside 40–79 |

Competing deaths are labelled as no event; rows censored early for other reasons are dropped.

## 4. Method

1. Participant-level 80/20 split (test set opened once), grouped and stratified.
2. Grouped 5-fold CV tuning on log-loss with the one-standard-error rule; out-of-fold Platt
   recalibration; unweighted ensemble; served model chosen on CV only.
3. Baselines: published FRS/PCE, the same recalibrated, and the same formula refitted.
4. Evaluation: AUROC, AUPRC, Brier, calibration intercept/slope, E/O, decision curves,
   subgroups; participant-level bootstrap with paired differences; nested CV over the whole
   cohort; input-group ablation.

## 5. Results

See README.md for the full tables. In short:

* **Framingham:** discrimination tie with the FRS (nested CV +0.003, CI −0.003 to +0.009);
  large calibration gain (E/O 1.48 → 1.00; Brier −0.007, CI excludes 0); restricting to
  the FRS's own inputs costs nothing.
* **NHANES:** ensemble beats the PCE (test +0.020, p = 0.03); nested CV +0.024 (CI +0.013 to +0.035), and +0.031 over the refitted PCE;
  using only the PCE's inputs costs 0.021–0.031 AUROC in CV.
* No model overfits materially: train-to-test AUROC drops are at most 0.03.

## 6. System

Flask REST API (`/models`, `/schema`, `/predict`, `/batch`, `/evidence`) serving both
calculators; React SPA with patient and clinician portals, evidence dashboard, PDF report,
CSV batch scoring; Docker image and Render blueprint; pytest suite.

## 7. Limitations and future work

American cohorts only; NHANES outcome is fatal events only; family history from 2005;
no survey weights; associations not causal effects. Future: external validation in a
non-US cohort, competing-risk survival models, the PREVENT equations as a third baseline,
and fairness analysis by race and income.
