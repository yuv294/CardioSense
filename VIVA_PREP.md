# CardioSense — Viva Preparation

Everything an examiner is likely to ask, with answers in plain language you can
say out loud. The rubric line is: *"Strong conceptual understanding and sound
technical justification."* That means you need to explain **why** you made each
choice, not just what the code does. Read this until you can say each answer in
your own words.

---

## 30-second project summary (memorise this)

"CardioSense predicts a person's 10-year risk of coronary heart disease from
routine health information — age, blood pressure, cholesterol, smoking, and so
on. We trained four machine-learning models and an ensemble on the Framingham
Heart Study data, compared them honestly against the established clinical risk
calculator, and served the best-calibrated model through a web app with two
portals — one for patients, one for clinicians."

---

## The questions, grouped

### A. Problem & data

**Q: What problem does this solve?**
It estimates the *future* risk of heart disease so people and doctors can act
early. It's a screening/prevention tool, not a diagnosis.

**Q: Why the Framingham dataset?**
It's a real, well-known epidemiological cohort — 4,240 people followed over
time, with a genuine 10-year outcome. Crucially, the inputs are things known at
a routine visit, so the model predicts the future rather than reading a test
result that already contains the answer.

**Q: Why is predicting a future event different from the usual heart-disease
project?**
Most student projects use datasets where the inputs include test results (like
exercise-test ST depression) and the "prediction" is really just detecting
disease already present — so they hit 90%+ accuracy that means little. Ours
predicts an event up to ten years away from basic risk factors, which is
genuinely harder and more useful. Lower headline numbers, far more honest.

**Q: How many features, and what did you add?**
15 raw inputs, 8 engineered features (pulse pressure, mean arterial pressure,
BP×age, a smoking-load proxy, cholesterol×age, a hypertension-AND-diabetes flag,
a high-glucose flag, age-squared) and 8 "Framingham-structure" features: ln(age),
ln(SBP), ln(BMI), male×ln(age), male×ln(SBP), male×smoker, male×diabetes and
treated-BP×ln(SBP). The last group copies the *form* of the clinical equation
(logs, separate effects for men and women, a different pressure term when
treated), which a plain linear model cannot express. It is the only addition that
measurably helped (+0.004 AUROC in repeated cross-validation).

**Q: How did you handle missing data?**
Median imputation for continuous fields, mode for categorical — but *fitted on
the training set only* (see leakage below). About 9% of glucose values were
missing, plus smaller amounts elsewhere.

---

### B. The most important concept: no data leakage

**Q: What is data leakage and how did you prevent it?**
Leakage is when information from the test set sneaks into training, making the
model look better than it really is. The classic mistake is scaling or imputing
on the whole dataset *before* splitting — then the training pipeline has secretly
"seen" the test rows' statistics. We split **first**, locked the test set away,
and fit every transformer (imputer, scaler) on the training rows only. The test
set is transformed using training statistics and opened exactly once, at the end.

**Q: Why does that matter for the grade / for science?**
Because a leaked model reports a number you can't trust. Our whole project is
built to produce an *honest* number, and this is the foundation of that.

---

### C. Models

**Q: Which models, and why those?**
- **Logistic Regression** — simple, fully interpretable linear baseline.
- **Random Forest** — bagged decision trees, robust, handles non-linearity.
- **HistGradientBoosting** — boosted trees, usually strongest on tabular data.
- **MLP (neural network)** — the deep-learning paradigm, for comparison.
- **Stacking ensemble** — combines all four via a logistic meta-learner.

Two paradigms (classical ML and deep learning) plus an ensemble — a proper
comparison, not one model.

**Q: Why not just use the neural network / deep learning?**
On tabular data this size, neural nets usually *don't* beat trees or even
logistic regression — and ours confirmed that. Deep learning shines on images,
text, and huge datasets, not 4,000 rows of tabular risk factors. Knowing when
*not* to use a tool is part of the judgement.

**Q: How did you tune them?**
5-fold stratified cross-validation on the training set only, grid-searching
hyperparameters, optimising **average precision** (not accuracy — see below).
The test set played no part in tuning.

**Q: What is stacking?**
It trains the four base models, then a final "meta" model learns how to best
combine their predictions. The meta-learner is trained on cross-validated
out-of-fold predictions so no model sees its own training rows — again, no leak.

---

### D. Evaluation — the metrics (examiners love this area)

**Q: Why not use accuracy?**
Because only 15% of people developed CHD. A lazy model that predicts "no disease"
for everyone scores 85% accuracy while being useless. Accuracy is misleading on
imbalanced data.

**Q: So what did you use instead?**
- **AUROC** — ranking skill: given one sick and one healthy person, how often
  does the model score the sick one higher? 0.5 = chance, 1.0 = perfect. Ours ≈ 0.70.
- **AUPRC** (average precision) — precision across recall levels, which respects
  the 15% base rate. Our main tuning metric.
- **Brier score** — calibration error: are the predicted percentages truthful?
  Lower is better.
- **Bootstrap 95% confidence intervals** — so we don't over-read a 0.01 gap as
  a real difference.

**Q: What is calibration and why did you care?**
Calibration asks: when the model says "20%", do about 20 out of 100 such people
actually develop disease? A model can rank people correctly yet state wrong
percentages. Since our app shows a real percentage to a person, calibration
matters as much as ranking. We applied **isotonic calibration**, which cut the
Brier score from 0.21 to 0.12 — a big improvement in how truthful the numbers are.

**Q: How did you pick the decision threshold?**
Not at the default 0.5. We chose it on the *training* out-of-fold predictions to
hit ~70% sensitivity (catch most future cases), then measured what that gave on
the test set: ~61% sensitivity at ~66% specificity (threshold 15.5%). Choosing the threshold on
training data, not test data, keeps the evaluation honest.

---

### E. The result — and how to defend the "tie"

**Q: Did your model beat the Framingham clinical score?**
Not clearly — and that's an honest, defensible finding, not a failure. On the
locked test set our served model scored AUROC 0.695 and Framingham 0.703; the
paired difference is -0.006 (95% interval -0.024 to +0.013), so statistically
they're tied. In repeated cross-validation on the training rows we are slightly
ahead (0.734 vs 0.721, +0.013, interval +0.003 to +0.023), because we gave the
model the structure of the Framingham equation. The two views differ by about
the sampling error of the test split, so the fair summary is "a tie to a small
edge". Where we are clearly better is calibration: the published score
under-predicts risk in this cohort (mean 12.5% vs 15.2% observed; 9.1% vs 12.8%
in women) and our calibrated model does not.

**Q: Why couldn't you beat it? Isn't that bad?**
The Framingham score was *derived from the Framingham study* — the same data
family we test on. So it's playing on home turf; it's close to the ceiling here.
Beating it would require either a different dataset or overfitting, and
overfitting would be cheating. Matching a decades-refined clinical gold standard,
using only risk factors and honest methodology, is a legitimate and strong
result. The research literature reports the same thing: ML rarely beats
well-built clinical scores on standard tabular risk factors.

**Q: Then what's the contribution of your project?**
Three things: (1) a rigorous, leak-free, calibration-aware evaluation pipeline;
(2) an interpretable, deployed tool with per-patient explanations; (3) a clear
demonstration that ML *matches* the clinical standard, which is useful evidence
in itself.

---

### F. Explainability

**Q: How does the app explain a prediction?**
For the linear served model, each feature's contribution is exactly its
coefficient times the standardised input value — a signed push on the risk. We
show which factors pushed risk up and down, and by how much. For the clinician
portal we show the full breakdown and all models' probabilities.

**Q: Is this SHAP?**
For a linear model, this signed-contribution method *is* what SHAP reduces to,
exactly and without approximation — so we compute it directly and avoid the extra
dependency. We can explain that choice if asked.

---

### G. The application

**Q: Describe the architecture.**
Three tiers: a scikit-learn model layer producing saved artifacts; a Flask REST
API serving predictions, schema, metrics, and explanations; and a React
front-end with a patient portal and a clinician portal. The frontend talks to
the backend over HTTP/JSON.

**Q: Why two portals?**
Patients need a simple risk number, plain-language factors, and a report.
Clinicians need more: every model's probability, model agreement, full feature
contributions, subgroup context, and the explicit decision threshold. Same
engine, two levels of detail.

**Q: Is this a medical device?**
No, and we state that clearly throughout. It's an educational tool trained on a
historical cohort, not clinically validated, and not for diagnosis or treatment.
Being upfront about limitations is part of doing this responsibly.

---

### H. Likely "gotcha" questions

**Q: What's the biggest weakness of your project?**
It's trained on one historical cohort, so it may not transfer to other
populations — especially non-Western or modern ones. External validation on an
independent dataset is the clear next step.

**Q: If you had more time, what would you do?**
External validation on a second dataset; add HDL to use the full laboratory
Framingham model; a fairness audit across sex and age subgroups.

**Q: Why should we trust a model that only ties the baseline?**
Trust comes from the *process*, not a single number: no leakage, honest metrics,
calibrated probabilities, confidence intervals, and transparent limitations. A
model that honestly ties is more trustworthy than one that claims to win through
a hidden leak.

---

## One-line answers to have ready
- Dataset: Framingham, 4,240 people, 10-year CHD outcome.
- Split: 80/20 stratified, test set locked.
- Models: LogReg, RF, HistGradientBoosting, MLP, + stacking.
- Best: calibrated logistic regression on 31 features, test AUROC 0.695.
- Baseline: Framingham Risk Score, test AUROC 0.703 — statistical tie (paired difference -0.006, CI -0.024 to +0.013).
- Key metrics: AUROC, AUPRC, Brier, calibration curve, bootstrap CIs.
- Why not accuracy: 15% imbalance makes it misleading.
- Headline: ML matches the clinical gold standard with honest methodology.

---

## Added questions (likely follow-ups)

**Q: How did you decide which model to serve? Did you use the test set?**
No. The served model is picked from cross-validation on the training set only.
The stack has the best cross-validated average precision (0.367). It has no
standard error of its own, so we use the average of the base models' (0.022);
logistic regression (0.357) and the MLP (0.366) are within noise of it, boosting
(0.339) and the forest (0.336) are not. The one-standard-error rule says: among
models within one standard error of the best, take the simplest.
That is logistic regression, which is also the only model whose explanations are
exact. The test set made no decision; it only produced the final numbers.

**Q: Isn't the Framingham score built for a different outcome?**
Yes, and we say so. The office-based D'Agostino score predicts general
cardiovascular disease (including stroke and heart failure), while our label is
coronary heart disease. That probably handicaps the baseline a little, so "tie"
is a fair reading, not "we are as good as the gold standard".

**Q: Your threshold was meant to give 70% sensitivity but you got 61%. Why?**
The threshold was tuned on out-of-fold training predictions, which are slightly
optimistic, and the test set has only 129 events so sensitivity is noisy (a wide
interval). We report the observed 61% rather than adjusting the threshold on the
test set, which would be leakage.

**Q: Is the login secure?**
No, and we state that. Accounts live in the browser with hashed passwords. It
separates the two portals and keeps each person's records apart on a device.
Real security needs server-side authentication and a database.

**Q: What does the Model performance page show?**
How the models did on 848 held-out patients: AUROC for every model against the
Framingham score, the calibration curve, and the confusion matrix at the 15%
screening threshold. It uses no patient input; it is the evidence behind the
model.

**Q: Why do the five models disagree on one patient?**
They are different learners fit on a small dataset, so they extrapolate
differently for unusual combinations. The spread is shown to clinicians as a
signal of uncertainty. Only the calibrated logistic regression drives the
headline number; the others are shown raw for comparison.

**Q: What would you do with more time?**
External validation on another population, more predictors (HDL/LDL, family
history), time-to-event modelling, a second untouched test set, and real
server-side accounts. We looked for the extra data: the BioLINCC Framingham
teaching dataset has HDL/LDL and follow-up time but no family history and must be
requested; NHANES with CDC linked mortality is free but its outcome is death, not
CHD. Neither could be obtained and validated in time, so it is future work and we
did not invent those columns.

**Q: Where is the model code, and how would you improve a model?**
Every model, its search grid and its "simplicity order" are in
`scripts/model_specs.py`; `train.py` runs the search. To change one I edit its
entry there, then re-run train, evaluate, explain, population, make_plots and
diagnostics, and restart the backend. I judge changes by cross-validated score
and the train-versus-CV gap, not by the test score.

**Q: Did your models overfit? What did you do about it?**
Yes at first: the random forest scored 0.877 on training data and 0.669 on test.
I widened the grids to include simple models, switched to the one-standard-error
rule (simplest setting within noise of the best), and removed the neural
network's accuracy-based early stopping. Train-to-test gaps fell from about 0.2
to 0.07, and test AUROC did not get worse. The remaining ~0.03 to 0.04 gap between CV and
test is mostly the test split being harder (Framingham, with nothing fitted, drops
0.018) plus sampling noise from 129 events.

**Q: Was the test set still untouched after that?**
Not strictly: the earlier configuration had been scored on it before the fix. The
fix was triggered by the train-versus-CV gap and chosen by cross-validation, and
nothing was changed after the final test results. We say so in the report and list
a second untouched test set as future work.

**Q: You tried to improve the model. What worked and what did not?**
We judged ideas on the training rows only, with 5×5 repeated cross-validation,
the penalty chosen inside each fold, and a paired bootstrap against Framingham
(`scripts/ladder.py`). Copying the Framingham equation's structure helped
(+0.004 AUROC over our old model, interval +0.001 to +0.009). Missing-value flags,
the Framingham score as a feature, the score as a fixed offset (worse), and
boosting did not help.

**Q: Did you change your selection rule after seeing results?**
Yes, and we disclose it. The rule written in advance (first step within one
standard error of the best average precision) picked the old model. The paired
AUROC comparison showed the structure features ahead with an interval excluding
zero, so we amended the rule to use that. The gain is small, the interval is not
adjusted for the several steps tried, and it is not confirmed on the test split.

**Q: Why does nothing get past about 0.73?**
`scripts/ceiling.py`: age, sex, systolic BP, smoking and diabetes already reach
0.732 and the other 26 features add nothing; the learning curve is flat from
about 2,000 patients. The limit is the information in the inputs.

**Q: Is the model equally good for everyone?**
No. `scripts/subgroups.py`: calibration slopes are 0.72 in women and 0.47 in
people aged 60 and over (1 is ideal; lower means over-spread predictions), and
AUROC within an age band is only 0.61 to 0.67. Decision-curve net benefit at a 15%
threshold is 0.053 (Framingham as published 0.047, flagging everyone 0.002).
