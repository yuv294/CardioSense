# CardioSense — Project Synopsis (revised)

**Submitted by:** Yuvika Das · Rajapurohitham Sahana · Rosanne Chugh

## Abstract

Cardiovascular disease is a leading cause of death worldwide, and identifying
at-risk individuals early allows preventive intervention. Established clinical
risk calculators are rule-based and were derived from specific cohorts, and may
mis-estimate risk in populations unlike their derivation sample. CardioSense
develops and fairly compares several machine-learning models — an interpretable
Random Forest, a gradient-boosted ensemble, a neural network, and a logistic-
regression reference — for predicting an individual's 10-year risk of coronary
heart disease (CHD) from routinely available clinical attributes. All models are
benchmarked against a published rule-based score (the Framingham Risk Score), and
the best-calibrated model is delivered through a Flask REST API and a React web
interface. The system is trained and validated on the Framingham Heart Study
teaching cohort (4,240 participants).

## 1. Introduction

### 1.1 Overview
CardioSense is an end-to-end machine-learning application for 10-year CHD risk
estimation. It spans data preprocessing, model training, honest evaluation
against a clinical baseline, and deployment as a clinician- and patient-facing
web application.

### 1.2 Motivation
Cardiovascular disease accounts for roughly a third of global deaths. Estimating
an individual's future risk from information available at a routine visit — age,
sex, blood pressure, cholesterol, smoking, diabetes, BMI — lets people and
clinicians act before an event occurs. The question this project asks is whether
modern machine learning improves on the rule-based calculators already in use.

### 1.3 Problem Statement
Rule-based cardiovascular risk calculators are limited by the cohorts and eras
they were derived from, and clinician-facing tools that make a personalised
estimate actionable are not always available. There is a need for an ML pipeline
that (i) predicts individual 10-year CHD risk from routine attributes, (ii) is
honestly benchmarked against an established clinical score, and (iii) is
delivered through an accessible web interface with transparent, calibrated
output.

### 1.4 Objectives
- Preprocess the Framingham cohort with a leak-free, split-first pipeline.
- Train and tune four models spanning interpretable and deep-learning paradigms.
- Implement the Framingham Risk Score as a rule-based baseline.
- Evaluate all models on a locked test set using discrimination (AUROC, AUPRC),
  calibration (Brier score, calibration curve), and bootstrap confidence intervals.
- Deploy the best-calibrated model via a Flask API and a React web app with
  per-prediction explanations and a downloadable report.

## 2. Literature Review

Machine learning for cardiovascular risk prediction has been studied widely.
Comparative studies (e.g. against the Framingham and ACC/AHA pooled-cohort
equations) report that ML models often match, but do not consistently beat,
well-constructed clinical risk scores on standard tabular risk factors —
particularly once calibration, not just discrimination, is considered. Random
Forests (Breiman, 2001) and gradient-boosted trees are strong tabular baselines;
recent work questions whether neural networks offer an advantage on tabular
clinical data. This motivates a careful, calibration-aware comparison rather than
an accuracy-only claim.

## 3. Proposed Methodology

### 3.1 Pipeline
Data preprocessing → model training (CV-tuned) → evaluation against the
rule-based baseline on a locked test set → deployment.

### 3.2 Data Preprocessing
- Stratified 80/20 split performed **first**; the test set is locked.
- Imputation (median/mode) and standardisation are fit on the training set only
  and applied to the test set, preventing leakage.
- Defaults derived from training data are stored so the live API fills missing
  optional inputs consistently and discloses every assumed field.

### 3.3 Models
Logistic Regression, Random Forest, HistGradientBoosting, and an MLP neural
network, each tuned with 5-fold stratified cross-validation on the training set,
optimising average precision to respect the ~15% class imbalance, plus a stacked
ensemble. Overfitting is controlled in the code: each grid includes very simple
settings and the simplest setting within one standard error of the best is chosen.

### 3.4 Rule-Based Baseline
The Framingham General CVD risk score (office/BMI model, D'Agostino 2008) is
computed directly from the cohort attributes and scored on the same locked test
set, giving a like-for-like benchmark.

### 3.5 Evaluation
AUROC, AUPRC, Brier score, calibration curves, and 2,000-sample bootstrap
confidence intervals. The served model is isotonically calibrated so its output
is a trustworthy probability. An operating threshold is selected on training
out-of-fold predictions for a target sensitivity.

### 3.6 System Architecture (three-tier)
- **Model layer:** scikit-learn pipeline producing serialized artifacts.
- **Backend:** Flask REST API (`/predict`, `/schema`, `/metrics`, `/health`).
- **Frontend:** React SPA — risk form, dashboard, history, learning centre,
  and a client-side PDF report.

## 4. Tools & Dataset

Python, scikit-learn, Flask, React (Vite, Tailwind), Recharts, jsPDF. Dataset:
Framingham Heart Study teaching cohort — 4,240 participants, 15 predictors,
binary 10-year CHD outcome.

## 5. Results (locked test set, n = 848)

No ML model clearly beat the Framingham baseline; the served model (calibrated
logistic regression on 31 features, AUROC 0.695) and the rule-based score (AUROC
0.703) have overlapping confidence intervals (paired difference −0.006, 95% CI
−0.024 to +0.013). In repeated cross-validation on the training data the model is
slightly ahead (+0.013, CI +0.003 to +0.023), the only gain coming from giving it
the structure of the Framingham equation; other ideas (missing flags, the score as
a feature or offset, boosting) gave nothing, and an ablation shows the ceiling of
about 0.73 AUROC is set by the inputs. Calibration improved the served model's
Brier score (0.212 → 0.123), and the published score under-predicts risk in this
cohort. This tie-to-small-edge result is consistent with the literature and is
reported transparently.

## 6. Expected Outcomes & Applications

A reproducible, leak-free pipeline; four tuned models with an honest clinical
benchmark; a calibrated, explainable risk estimator; and a deployed web
application. Applicable to preventive-health screening and as a teaching example
of rigorous model evaluation.

## 7. Future Scope

External validation on an independent cohort; incorporation of additional
predictors (HDL/LDL, family history; candidate public sources are the BioLINCC
Framingham teaching dataset and NHANES with CDC linked mortality); time-to-event
modelling; and recalibration for non-Framingham populations.

## 8. Limitations

Not clinically validated; trained on a single historical cohort; optional inputs
imputed when blank; browser-local history only; a risk estimate is not a
diagnosis.
