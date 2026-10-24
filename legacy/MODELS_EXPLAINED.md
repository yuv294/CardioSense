# CardioSense: every model explained

This is the document to study for the viva. For each model it answers the same questions:

1. What is it, in plain words?
2. How does it actually work inside?
3. How did we build and train it, with the exact settings we used?
4. What results did it get?
5. Why did it get those results, and why did it not do better?

Every number here comes from our own final run (after the overfitting fix in Part 10 and the improvement ladder in Part 15). Nothing is borrowed.

---

## Part 0. Where the code for every model lives

| File (in `scripts/`) | What is in it |
|---|---|
| `model_specs.py` | **The definition of all four base models**: the scikit-learn estimator, its fixed settings, the grid of settings to search, and the rule that says which setting is "simpler". This is the file to edit to change or improve a model. |
| `train.py` | Runs the grid search with 5-fold cross-validation, applies the one-standard-error rule, refits the winners, builds the stacked ensemble, saves `artifacts/model_*.joblib` and `artifacts/cv_results.json`. |
| `evaluate.py` | Chooses the model to serve (from CV only), calibrates it, picks the threshold, then scores everything on the test set once. |
| `diagnostics.py` | Prints train, cross-validated and test AUROC for every model and the model internals (tree depth, weights, epochs). This is the overfitting check. Saves `reports/diagnostics.json`. |
| `explain.py`, `population.py`, `make_plots.py` | Feed the app and the figures. |
| `features.py` | The 8 engineered and 8 Framingham-structure features, shared by training and the API. |
| `ladder.py`, `offset_logreg.py` | The improvement ladder (Part 15): repeated, nested cross-validation with paired comparisons; the custom offset logistic regression. |
| `subgroups.py`, `ceiling.py`, `paired_test.py` | Calibration and net benefit by subgroup; input ablation and learning curve; paired test-set comparison with Framingham. |

**To change a model:** edit its entry in `model_specs.py` (a wider grid, a new setting, or a different estimator), then, from the `scripts/` folder, run in this order:

```
python train.py          # tune and refit (a few minutes)
python evaluate.py       # choose, calibrate, score
python explain.py        # refresh the app's explanations
python population.py     # refresh the cohort comparison
python make_plots.py     # refresh the figures
python diagnostics.py    # train / CV / test gap for every model
python ladder.py         # improvement ladder (about 10 minutes)
python subgroups.py      # subgroup calibration and decision curves
python ceiling.py        # ablation and learning curve
python paired_test.py    # paired test-set bootstrap vs Framingham
```

Then restart the backend so the app loads the new models.

**One rule for improving honestly:** judge every change by the cross-validated score printed by `train.py` (and the train-versus-CV gap from `diagnostics.py`), not by the test score. Each time you change something because of the test score, the test set stops being an independent check.

Ideas that are worth trying (in order of likely value): new input information (family history, LDL/HDL cholesterol; see Part 15.6 for where to get it); a wider grid with values below the current edge (for example `C` below 0.001); elastic-net logistic regression (`penalty="elasticnet"`, `solver="saga"`); a second held-out test set for a stricter final check. For judging small improvements use `ladder.py` (repeated, nested cross-validation with paired intervals), not the test set.

---
## Part 1. The ideas you need first

### 1.1 What the models are asked to do

Each patient is a row of 31 numbers (15 measured, 8 calculated from them, and 8 more that copy the form of the Framingham equation; Part 15). The answer for each patient is 0 (no coronary heart disease within ten years) or 1 (disease). This is **binary classification**.

We do not just want 0 or 1. We want a **probability**, such as "39%", because the app shows a percentage to a person. All five models can output a probability.

### 1.2 What "training" means

A model has settings inside it, called **parameters** (for example, the weight given to age). Training means adjusting the parameters until the model's predictions match the known answers as closely as possible on the **training set** (3,392 patients).

There are two separate kinds of settings:

- **Parameters** are learned from data (the weight on age).
- **Hyperparameters** are chosen by us before training (how many trees, how strong the penalty). We choose these by trying several values and checking which works best on data the model was not trained on. That is **hyperparameter tuning**.

### 1.3 Overfitting and underfitting (the most important idea)

- **Overfitting**: the model memorises quirks of the training patients that will not repeat. It looks brilliant on training data and worse on new patients.
- **Underfitting**: the model is too simple or too restricted to capture real patterns. It does poorly everywhere.

We can see which one is happening by comparing three scores for the same model:

| Score | Meaning |
|---|---|
| Train AUROC | How well it does on the data it learned from |
| CV AUROC | How well it does on training patients it was **not** trained on (cross-validation) |
| Test AUROC | How well it does on the locked test set |

A large drop from train to CV means overfitting. We measured all three for every model (Table in Part 9).

### 1.4 Cross-validation (how we tuned without touching the test set)

5-fold stratified cross-validation splits the training set into five equal parts. We train on four parts and score on the fifth, rotate five times, and average. Each patient is used for scoring exactly once, by a model that never saw them. "Stratified" keeps the 15% disease rate in every part.

We tried each combination of hyperparameters this way and chose among them with the one-standard-error rule (Part 10), which prefers the simplest setting that is within noise of the best. The test set was not involved.

### 1.5 The class imbalance problem

Only 15.2% of patients developed disease (515 of 3,392 in training). A lazy model that says "no disease" for everyone is right 85% of the time and useless.

We handled this in two ways:

- **Balanced class weights** (logistic regression, random forest, stacking meta-learner). Each patient who had disease counts **3.29 times** in the loss, and each healthy patient counts **0.59 times**. The formula is `n / (2 x class count)`: 3,392 / (2 x 515) = 3.29 and 3,392 / (2 x 2,877) = 0.59. The model is punished more for missing a sick patient.
- **Average precision as the tuning score**, instead of accuracy. It measures how well the model finds the rare positives.

A side effect: balanced weights push predicted probabilities upward, so the raw outputs are too high. That is why we calibrate (Part 7).

### 1.6 What the inputs look like

All continuous inputs are standardised (training mean subtracted, divided by training standard deviation). This puts age (about 50) and cholesterol (about 237) on the same scale. It matters a lot for logistic regression and the neural network, and not at all for trees.


---

## Part 2. Model 1: Logistic regression (the one we serve)

### What it is
A weighted score that is turned into a probability. It is the standard model in medicine, and the Framingham score is itself a logistic-type model.

### How it works
1. Multiply each input by a weight and add them up, plus an intercept. This gives a **log-odds** score:
   `z = intercept + w1 x age + w2 x sysBP + w3 x smoker + ...`
2. Convert to a probability with the **sigmoid** function: `p = 1 / (1 + e^(-z))`. This squashes any number into the range 0 to 1. A large positive z means near 100%, large negative near 0%, and z = 0 means 50%.
3. **Training** finds the weights that make the predicted probabilities as close as possible to what happened. Technically it minimises the **log-loss**, which heavily punishes confident wrong answers.

### The regularisation penalty (L2) and the parameter C
Without a penalty, with 31 inputs and some of them correlated (age appears in seven of our features), the weights can become large and unstable, chasing noise. **L2 regularisation** adds a cost for large weights, so the model prefers small, spread-out weights.

`C` controls this and is the **inverse** of the strength: **small C = strong penalty**.

### How we built it
- Definition in `model_specs.py`: `LogisticRegression(max_iter=2000, class_weight="balanced")`, L2 penalty, `lbfgs` solver (converged in 10 iterations).
- Grid: `C` in {0.001, 0.003, 0.01, 0.03, 0.1, 1.0}: 6 values x 5 folds = 30 fits.
- **Selected: C = 0.001.** Here is the honest story. All six values of C are within one standard error (0.020) of the best, so the data cannot tell them apart: the best-scoring value (C = 0.01) has an average precision of 0.363 and the one we picked 0.357. The one-standard-error rule then takes the simplest, the strongest penalty. Be ready to say that C = 0.001 sits at the edge of the grid and that this is a limitation.
- With such a strong penalty the weights are small and roughly follow how strongly each feature is associated with the outcome. That is fine for ranking patients. The **scale** of the probabilities is then repaired by calibration (Part 7).

### What it learned
Largest standardised weights: log age 0.112, age 0.108, age squared 0.103, blood pressure x age 0.098, cholesterol x age 0.083, smoking load 0.079, glucose 0.076, male x log age 0.063, cigarettes per day 0.062, male x log systolic BP 0.061. These match medical knowledge, which is a good sanity check. Because the inputs are standardised, weights are comparable: a bigger weight means a bigger influence per standard deviation. The app adds up related features (all the age terms count as "Age") before showing a patient their top factors.

### Results
| | AUROC | AUPRC |
|---|---|---|
| Train | 0.739 | |
| 5-fold CV | 0.734 | 0.357 (AP) |
| Test | 0.695 calibrated (CI 0.65 to 0.74); 0.697 before calibration | 0.292 |

### Why it did well
- Few effective parameters and a strong penalty: train 0.739 vs CV 0.734, a gap of only 0.005.
- Risk-factor relationships are close to straight lines on the log-odds scale (more blood pressure, more risk), which is what the model assumes.
- Our engineered features (age squared, BP x age, log age, sex-specific terms and so on) let this linear model capture curved and interacting effects.
- With only 515 positive cases there is not enough data for a flexible model to do better.

### Why it did not beat Framingham
It is essentially the same kind of model fitted to similar variables. The Framingham coefficients were derived from this very study population over decades of follow-up, so there is little room to improve on it from 3,392 patients. The gap on the test split (0.695 vs 0.703; paired difference -0.006 with interval -0.024 to +0.013) is far smaller than the uncertainty. In repeated cross-validation on the training rows it is actually slightly ahead (0.734 vs 0.721, difference +0.013, interval +0.003 to +0.023), mostly because we gave it the structure of the Framingham formula (Part 15). Either way: a tie to a small edge, not a clear win.

### Its weakness
It cannot find a pattern that is not linear in its inputs unless we supply it (which is why we engineered features).

---

## Part 3. Model 2: Random forest

### What it is
A large committee of decision trees that vote.

### How a single decision tree works
A tree asks a series of yes/no questions: "Is age above 54? If yes, is systolic pressure above 140? ..." Each question splits the patients into two groups. It picks, at each step, the question that best separates sick from healthy, measured by **Gini impurity** (how mixed a group is: 0 means all one class). It keeps splitting until it reaches the end, and each final group is a **leaf**. A new patient follows the questions down to a leaf and gets the share of sick patients in that leaf as their probability.

A single deep tree **overfits badly**: it memorises the training patients. That is where the "forest" idea comes in.

### What makes it a random forest
1. **Bagging**: each tree is trained on a random sample of the patients, drawn with replacement (a **bootstrap sample**). Each tree sees a different version of the data, about 63% unique patients.
2. **Random feature subsets**: at every split, a tree is only allowed to consider a random handful of features (5 of our 31, the square root). This stops all trees from picking the same strong feature first and makes them different from each other.
3. **Averaging**: 400 trees each give a probability and we average them. Errors that are random cancel out. This reduces **variance** (instability) while keeping the flexibility.

### How we built it
- Definition in `model_specs.py`: `RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample")`.
- Grid: depth {3, 5, 8, none} x minimum patients per leaf {20, 40, 80} x features per split {sqrt, 40%} = 24 combinations x 5 folds = 120 fits.
- **Selected: depth 3, 80 patients per leaf, sqrt features.** All 24 settings were within one standard error of the best, so the rule took the simplest. The "best score" setting would have been depth 5 with 20 per leaf.
- `balanced_subsample` applies the class weights separately within each tree's bootstrap sample.
- The final trees are 3 levels deep with about 7.5 leaves each (the first version: about 12 levels, 74 leaves).

### What it learned
Impurity-based feature importance: BP x age 0.188, log age 0.114, age squared 0.113, age 0.112, cholesterol x age 0.090, log systolic BP 0.054, systolic BP 0.048, pulse pressure 0.048. Impurity importance is biased towards continuous features (they offer many split points), which is why a forest's ranking differs from the logistic regression's.

### Results
| | AUROC | AUPRC |
|---|---|---|
| Train | 0.747 | |
| 5-fold CV | 0.722 | 0.336 (AP) |
| Test | 0.680 (CI 0.63 to 0.73) | 0.297 |

### Why it did not do better
After the fix it no longer overfits much (train 0.747, CV 0.722; before: 0.877 and 0.726). But it is the weakest model. Trees can only make step-like, axis-aligned splits, and with trees only 3 levels deep it cannot represent a smooth risk that rises steadily with age and pressure as well as a weighted sum does. Heart disease ten years out is also **partly random**, which caps every model.

### Another weakness: probabilities
A forest's probability is the average of its trees' votes, which is not a calibrated chance, and our class weights inflate it further. Its raw Brier score is 0.213 (worse than the 0.129 you get by always guessing the base rate), so it must not be shown to a person without calibration.

---

## Part 4. Model 3: Gradient boosting (HistGradientBoosting)

### What it is
A committee of small trees, like the forest, but built **one after another**, each one fixing the mistakes of those before.

### How it works
1. Start with a simple guess (the overall 15% rate for everyone).
2. Measure how wrong the guess is for each patient (the gradient of the loss, in practice the "error signal").
3. Train a **small** tree whose job is to predict that error.
4. Add a **fraction** of that tree's output to the current prediction. The fraction is the **learning rate**.
5. Repeat. Each new tree corrects what is still wrong.

This is gradient descent, but moving through the space of functions instead of through numbers. The forest uses many independent deep trees averaged; boosting uses many shallow trees added in sequence.

"**Histogram**" refers to a speed trick: continuous values are first sorted into up to 255 bins, so finding the best split is fast.

### How we built it
- Definition in `model_specs.py`: `HistGradientBoostingClassifier(max_iter=300, early_stopping=True, n_iter_no_change=15)`, loss `log_loss`.
- Grid: learning rate {0.03, 0.06} x depth {2, 3, 5} x leaves {7, 15} x minimum per leaf {20, 50, 100} x L2 penalty {0, 5} = 72 combinations x 5 folds = 360 fits.
- **Selected: learning rate 0.03, depth 2, 7 leaves, 100 per leaf, L2 = 0.** Sixty-three of the 72 settings were within one standard error of the best, so the rule took the simplest.
- **Early stopping**: 10% of the training data is held aside internally and training stops when its log-loss has not improved for 15 rounds. It stopped after **98 of 300 rounds**. (For boosting, early stopping watches the log-loss, which is a sensible signal; it is the MLP's version that watches accuracy.)
- A low learning rate means each tree makes only a small correction. This is a safeguard against overfitting.
- **We did not use class weights here.** Early on we passed sample weights through the grid search and the cross-validated score jumped to 0.71, which looked wonderful and was wrong: the weights were also applied to the validation folds. We noticed it was implausibly high and removed the weights. Good viva story about checking suspiciously good numbers.

### Results
| | AUROC | AUPRC |
|---|---|---|
| Train | 0.754 | |
| 5-fold CV | 0.720 | 0.339 (AP) |
| Test | 0.689 (CI 0.64 to 0.74) | 0.288 |

### Why it did not do better
First version (depth 5, 31 leaves, no penalty): train 0.830, CV 0.725, test 0.662. After the fix the train score is 0.754 and the test score rose to 0.689, so the fix helped. It still does not beat logistic regression because, with this much noise, the extra flexibility has little real structure to find. On a bigger dataset with more informative inputs boosting would usually win; here it does not.

---

## Part 5. Model 4: Neural network (multilayer perceptron, MLP)

### What it is
A layered network of small calculating units ("neurons"), loosely inspired by the brain. It is a very flexible function that can in theory learn almost any pattern.

### How it works
- **A neuron** takes inputs, multiplies each by a weight, adds a bias, and passes the result through an **activation function**. We used **ReLU**: it outputs the number if positive and 0 if negative. The activation is what lets the network learn curves; without it, stacking layers would just be one big linear model.
- **A layer** is many neurons side by side, each looking at all the outputs of the previous layer.
- Our network: 31 inputs, then one hidden layer of **16 neurons**, then 1 output neuron with a sigmoid that gives a probability. Written `(16,)`.
- **Parameters**: 31x16+16 + 16x1+1 = **529 weights and biases**, about one-sixth of the 3,392 training patients. (The first version had 11,393, more than three times the number of patients.)

### How it is trained
1. **Forward pass**: push a patient through the layers to get a predicted probability.
2. **Loss**: compare with the truth using log-loss.
3. **Backpropagation**: work backwards through the layers, using calculus (the chain rule), to find how much each weight contributed to the error.
4. **Optimiser**: nudge every weight slightly to reduce the error. We used **Adam**, which adapts the step size for each weight, with a starting rate of 0.001. Patients are processed in small batches (200 at a time) and one full pass through the data is an **epoch**.

### How we built it
- Definition in `model_specs.py`: `MLPClassifier(solver="adam", learning_rate_init=0.001, early_stopping=False)`.
- Grid: hidden layers {(8), (16), (32, 16)} x L2 penalty alpha {0.01, 0.1, 1.0} x epochs {30, 100} = 18 combinations x 5 folds = 90 fits.
- **Selected: (16), alpha = 1.0, 30 epochs.** Ten of the 18 settings were within one standard error; the rule took the simplest.
- `alpha` is an **L2 penalty** on the weights, the same idea as in logistic regression. The strongest value won.
- **Early stopping was removed on purpose.** scikit-learn's built-in early stopping watches validation **accuracy**. With 84.8% of patients healthy, accuracy barely moves, so in our first version training stopped after 14 epochs, before the network had learned. Now the number of epochs is simply a tuned setting, chosen by cross-validated average precision. This raised the cross-validated AP from 0.307 to **0.366**, the best of any model.
- scikit-learn's MLP does **not support class weights**, so it is trained on the natural 15%/85% mix. That is why its raw probabilities are not inflated (raw Brier 0.122).

### Results
| | AUROC | AUPRC |
|---|---|---|
| Train | 0.756 | |
| 5-fold CV | 0.735 | 0.366 (AP) |
| Test | 0.693 (CI 0.64 to 0.74) | 0.299 |

### Why it did well this time, and why we still do not serve it
- It now ties logistic regression on test AUROC (0.693 vs 0.695) and has the highest cross-validated AP of the four single models (0.366), so the earlier conclusion that "a neural network cannot work here" was partly an artefact of our early-stopping choice. With 16 neurons and a strong penalty it behaves almost like a smooth, lightly curved version of logistic regression.
- It is within one standard error of the other models, so it is not significantly better. And its prediction cannot be explained as exact signed contributions per feature, which is a core feature of the app. So we serve logistic regression and show the MLP's estimate in the clinician view.

---

## Part 6. Model 5: Stacked ensemble

### What it is
A model that combines the other four, learning how much to trust each.

### How it works
1. Take four base models: the tuned logistic regression, random forest, gradient boosting and MLP.
2. For each training patient, get a prediction from each base model **using a version trained without that patient** (5-fold out-of-fold predictions). This is crucial: if a base model predicted for patients it was trained on, its predictions would be unrealistically good and the meta-model would learn to trust it too much.
3. Train a **meta-learner** (a logistic regression, with balanced class weights) that takes the four predictions as its inputs and learns the best weighting.
4. At the end, refit the base models on all training data.

### What the meta-learner decided
Weights on the four models' probabilities: logistic regression **3.11**, neural network **2.36**, random forest **1.06**, gradient boosting **-0.75**. It leaned most on logistic regression and the neural network. The negative weight on boosting means it uses that model to cancel what it shares with the others, a sign that the four models carry nearly the same information.

### Results
| | AUROC | AUPRC |
|---|---|---|
| Train | 0.749 | |
| 5-fold CV | | 0.367 (AP) |
| Test | 0.696 (CI 0.64 to 0.74) | 0.299 |

### Why it did not do better
Stacking helps when the base models make **different** mistakes. Ours are all trained on the same 31 columns of the same 3,392 patients, so they make similar mistakes. Its test score (0.696) is no better than its best ingredient (0.697).

### Why we serve logistic regression
We apply a rule fixed in advance (Part 10): among models whose cross-validated average precision is within one standard error of the best, take the simplest. The stack is best (0.367) and has no standard error of its own, so we use the average standard error of the four base models (0.022): the band is 0.346 and above. Inside it are logistic regression (0.357), the MLP (0.366) and the stack; boosting (0.339) and the forest (0.336) are outside. The simplest inside is logistic regression. It is also the only one whose explanations are exact. The decision used no test-set information.

---

## Part 7. The Framingham Risk Score (the baseline) and calibration

### The score
It is a fixed formula, not a trained model. We use the office-based version from D'Agostino et al. (2008). For men:

`risk = 1 - 0.88431 ^ exp( 3.11296 x ln(age) + 0.79277 x ln(BMI) + 1.85508 x ln(SBP) + 0.70953 x smoker + 0.53160 x diabetes - 23.9388 )`

Worked example: a 55-year-old man, BMI 27, systolic pressure 140 (untreated), smoker, no diabetes:
- 3.11296 x ln(55) = 12.475; 0.79277 x ln(27) = 2.613; 1.85508 x ln(140) = 9.167; smoker 0.710; diabetes 0.
- Sum = 24.964. Subtract 23.9388 = 1.025. exp(1.025) = 2.788.
- Risk = 1 - 0.88431^2.788 = **29.0%**.

Because nothing is fitted to our data, the score has no training set. It scored 0.721 AUROC on our training rows and 0.703 on the test rows. **That 0.02 gap is a useful fact: our test split is slightly harder than our training split**, so part of every model's drop from CV to test is just the particular patients that landed in the test set.

### Calibration (making the percentages truthful)
Balanced class weights make a model over-state probabilities. Our raw logistic regression gave a Brier score of 0.212, much worse than guessing the base rate (0.129).

**Isotonic calibration** fixes this. It collects the model's out-of-fold scores on training patients and learns a monotonic (never decreasing) mapping from raw score to the **actual** observed frequency. For example, "patients who got raw 0.70 actually had disease 20% of the time, so map 0.70 to 20%". We fitted this with 5-fold cross-validation on training data only. The Brier score fell from 0.212 to **0.123**. This is chosen in advance, not by looking at test results.

### The screening threshold
A screening tool needs a "flag" cutoff. We took out-of-fold probabilities from the calibrated model on the training set and chose the highest threshold that still catches at least 70% of true cases: **15.5%**.

On the test set it caught 61% (not 70%). Two honest reasons: out-of-fold results are slightly optimistic, and with only 129 test cases the sensitivity estimate is noisy. We did not adjust the threshold on the test set, because that would be leakage.

---

---

## Part 8. The metrics, with our numbers

| Metric | Plain meaning | Our served model on test |
|---|---|---|
| **AUROC** | Pick one patient who got disease and one who did not. How often does the model give the sick one the higher risk? 0.5 is a coin flip, 1.0 is perfect. | 0.695 (Framingham 0.703) |
| **AUPRC** | How well does it find the rare positives without too many false alarms? A model with no skill scores about 0.152 (the base rate). | 0.292 |
| **Brier score** | Average squared error of the predicted probabilities. Lower is better. Always guessing 15.2% scores 0.129. | 0.123 |
| **Sensitivity** | Of the people who really got disease, what share were flagged? 79 / 129. | 61% |
| **Specificity** | Of the people who did not, what share were cleared? 476 / 719. | 66% |
| **Precision** | Of those flagged, what share really got disease? 79 / (79 + 243). | 25% |

**Bootstrap confidence interval.** We resample the 848 test patients with replacement 2,000 times, compute AUROC each time, and take the middle 95%. The interval for our model is 0.65 to 0.74 and for Framingham 0.66 to 0.75. All six intervals overlap almost completely, which is why we describe the difference as a tie and not a loss.

---

## Part 9. All models side by side

| Model | Train AUROC | CV AUROC | Test AUROC | Test AUPRC | Train minus test | Diagnosis |
|---|---|---|---|---|---|---|
| Framingham score | 0.721 (no fitting) | n/a | 0.703 | 0.299 | 0.018 | Fixed formula; shows the test split is slightly harder |
| Logistic regression (served) | 0.739 | 0.734 | 0.695 (0.697 uncalibrated) | 0.292 | 0.042 | Train and CV nearly equal |
| Stacked ensemble | 0.749 | n/a | 0.696 | 0.299 | 0.053 | Extra complexity not paid back |
| Neural network | 0.756 | 0.735 | 0.693 | 0.299 | 0.063 | Ties logistic regression after removing accuracy-based early stopping |
| Gradient boosting | 0.754 | 0.720 | 0.689 | 0.288 | 0.065 | Shallow trees, early stopping |
| Random forest | 0.747 | 0.722 | 0.680 | 0.297 | 0.067 | Depth 3; weakest, trees too coarse for smooth risk |

**Before and after the overfitting fix** (train AUROC, test AUROC, train minus test):

| Model | Before | After |
|---|---|---|
| Logistic regression | 0.738, 0.699, 0.039 | 0.739, 0.697, 0.042 |
| Random forest | 0.877, 0.669, **0.208** | 0.747, 0.680, 0.067 |
| Gradient boosting | 0.830, 0.662, **0.168** | 0.754, 0.689, 0.065 |
| Neural network | 0.720, 0.670, 0.050 | 0.756, 0.693, 0.063 |
| Stacked ensemble | 0.789, 0.687, 0.102 | 0.749, 0.696, 0.054 |

**Reading the remaining gap honestly.** Every model has a gap of 0.03 to 0.04 between cross-validated and test AUROC (0.031 to 0.042), including logistic regression, which barely overfits (train 0.739, CV 0.734). The Framingham formula, with nothing fitted, drops 0.018 from training rows to test rows, so the test split is about 0.02 harder, and with 129 test events a test AUROC has a standard error of about 0.025. So the remaining gap is mostly the test split and sampling noise, not overfitting. The real overfitting (the 0.2 gaps) is gone.

---

## Part 10. How we fixed overfitting (the story to tell)

**What we saw.** In the first round the random forest scored 0.877 on its own training data, 0.726 in cross-validation and 0.669 on test; gradient boosting 0.830, 0.725, 0.662. That train-versus-CV gap is visible *before* touching the test set, so it is a legitimate trigger to change the method.

**Why it happened.** Our grids contained only flexible models (depth 5 or unlimited, 31 leaves, 128-64 neurons) and we took whichever setting had the single highest cross-validated score. A flexible model with a noisy "best" score is usually just the luckiest one.

**What we changed in the code (all in `model_specs.py` and `train.py`):**
1. **Wider grids with a simple end**: strong penalties, depth 2-3 trees, big leaves, small networks.
2. **One-standard-error rule** (Hastie, Tibshirani and Friedman, *Elements of Statistical Learning*, section 7.10). Compute the standard error of the best setting's five fold scores. Among all settings whose mean score is within one standard error of the best, choose the simplest. Each model's grid has a written "complexity order" for this. The same rule is applied across models in `evaluate.py`.
3. **Neural network early stopping removed** (it watched accuracy) and replaced by a tuned number of epochs.
4. **`diagnostics.py`** now reports train, CV and test AUROC for every model so overfitting is measured, not guessed.

**What happened.** Gaps of 0.2 fell to 0.07. Test AUROC of the tree models and the MLP rose slightly (forest 0.669 to 0.680, boosting 0.662 to 0.689, MLP 0.670 to 0.693). Logistic regression moved from 0.699 to 0.697, which is well inside the noise. The ceiling did not move: no model clearly beats Framingham (0.703).

**Be upfront about this (viva):** the earlier configuration had been scored on the test set before we made the change, so the test set is not perfectly untouched. The change was prompted by the train-versus-CV gap and chosen by cross-validation, and nothing was adjusted after the final test results. A stricter protocol would use a second untouched test set, which is listed as future work.

---

## Part 11. How we got the results, step by step

1. `preprocess.py`: split first (80/20, stratified, seed 42). Fit imputer and scaler on the training rows only. Output: 3,392 training rows and 848 test rows, 31 columns (15 raw, 8 engineered, 8 Framingham-structure).
2. `baseline_frs.py`: implement the Framingham formula.
3. `train.py`: tune each of the four models by 5-fold cross-validation (30 + 120 + 360 + 90 fits) with the one-standard-error rule, then build the stack. Save the models and the cross-validated scores.
4. `evaluate.py`: choose the model to serve from the cross-validation scores (logistic regression), calibrate it, choose the threshold on training out-of-fold predictions, and score every model and the baseline on the test set. Compute bootstrap intervals.
5. `explain.py` and `population.py`: extract the exact contribution of each feature and the cohort percentiles for the app.
6. `make_plots.py`, `diagnostics.py`: figures and the overfitting table.
7. `ladder.py`, `subgroups.py`, `ceiling.py`, `paired_test.py`: the improvement ladder, subgroup and net-benefit analysis, the ceiling diagnostics and the paired test comparison (Part 15). These read only the training rows, apart from `paired_test.py`, which reports the locked test set once.

No tuning or selection step reads the test set. (`evaluate.py` once had a bug in which the stack, which has no standard error of its own, was given a margin of 0.005 and the MLP was served by mistake; the margin is now the mean standard error of the base models, which gives logistic regression.)

---

## Part 12. Why nothing beat the clinical score (the five reasons)

1. **The inputs are few and already well understood.** Age, blood pressure, smoking and diabetes explain most of what can be explained. A complex model has little hidden structure to find.
2. **The outcome is partly random.** Whether a particular person has a heart event in ten years depends on things we do not measure: genetics, diet, stress, luck. This sets a ceiling on any model, around 0.70 to 0.75 AUROC for this kind of data. We measured it for our data in Part 15.4: about 0.73.
3. **Not enough positive cases.** 515 positives in training is small for flexible models. They needed heavy restriction to stop overfitting, and even then did not exceed the simple model.
4. **The Framingham equations were derived from this same cohort**, from decades of follow-up. Our models learn from a subset, so they start at a disadvantage.
5. **The literature says the same.** A systematic review (Christodoulou et al., 2019) found no consistent benefit of machine learning over logistic regression for clinical prediction.

What would help: **new** information (more people of the same kind did not help in our learning curve), such as family history, LDL and HDL cholesterol, or inflammatory markers.

---

## Part 13. Likely viva questions on the models

**Which models did you use and why those?**
Logistic regression (linear, interpretable, the clinical standard), random forest (non-linear, robust), gradient boosting (usually strongest on tables), a neural network (to test deep learning), and a stack (to see if combining helps). They cover the main families, so the comparison is fair.

**Where is the code for the models and how would you change one?**
`scripts/model_specs.py` defines every model and its search grid; `train.py` tunes them. To change a model I edit its entry in `model_specs.py` and re-run `train.py`, `evaluate.py` and the other scripts in Part 0.

**Explain how a random forest works.**
Many decision trees, each trained on a random resample of the patients and allowed to consider only a random subset of features at each split. They vote and the average is the probability. The randomness makes the trees different, and averaging cancels their individual errors.

**What is the difference between random forest and gradient boosting?**
Forest: independent deep trees averaged in parallel (reduces variance). Boosting: shallow trees built one after another, each correcting the previous errors (reduces bias). Boosting is more powerful but more prone to chase noise.

**Which hyperparameters did you tune and what do they do?**
Logistic: C (inverse penalty strength). Forest: maximum depth, minimum patients per leaf, features per split. Boosting: learning rate, depth, leaves, minimum per leaf, L2 penalty. MLP: layer sizes, L2 penalty alpha, number of epochs. All tuned by 5-fold cross-validation on training data, scored by average precision.

**Your models overfitted at first. What did you do?**
Part 10. We saw a train-versus-CV gap (0.877 vs 0.726 for the forest), widened the grids to include simple models, and added the one-standard-error rule, which picks the simplest setting within noise of the best. The gap fell from about 0.2 to 0.07.

**What is the one-standard-error rule and why use it?**
Cross-validated scores are noisy. Among all settings within one standard error of the best, we pick the simplest, because the best score is partly luck and simple models generalise better. It is a standard rule from *The Elements of Statistical Learning*.

**Your best hyperparameters were often the strongest regularisation. What does that tell you?**
That the data is small and noisy and rewards simple models. For logistic regression all six values of C were inside one standard error, so the rule took the strongest, which sits at the edge of the grid; a stronger penalty or elastic-net could be tried as future work.

**The test AUROC is still about 0.04 below cross-validation. Is that overfitting?**
Mostly not. The Framingham formula, which is not fitted at all, also drops 0.018 on the test split, so the split is slightly harder, and with 129 events the test AUROC has a standard error of about 0.025.

**Why did the neural network do badly at first and better later?**
Our first version used scikit-learn's early stopping, which watches accuracy; with 85% of patients healthy it stopped after 14 epochs, too early. We replaced it with a tuned number of epochs and a much smaller network (529 weights). The cross-validated AP rose from 0.307 to 0.366 and the test AUROC from 0.670 to 0.693.

**Why does your logistic regression get the best result, if it is the simplest?**
It ties the others within noise; we serve it because the problem is small, noisy and close to linear, and its explanations are exact. Simple models are often best here, consistent with published evidence.

**What does the stacking ensemble do and why did it not help?**
It learns a weighted combination of the four models' predictions, trained on out-of-fold predictions so there is no leak. It did not help because all four models learn from the same data, so they make similar mistakes.

**Did you use the test set to pick the model?**
No. We chose by cross-validation with the one-standard-error rule. The test set reports the result. We disclose that an earlier configuration had been scored on it before the overfitting fix.

**How do you know the test result is not just luck?**
We computed 95% bootstrap confidence intervals. They show all models are statistically indistinguishable from the Framingham score on the test split, and a paired bootstrap of the test set gives a difference of -0.006 (interval -0.024 to +0.013), which is why we call it a tie.

**What is class imbalance and how did you deal with it?**
Only 15% of patients have the outcome. We used balanced class weights (positives count 3.29 times, negatives 0.59 times), tuned using average precision instead of accuracy, and calibrated the final probabilities.

**Why calibration?**
Because the app shows a percentage to a person. A model can rank people correctly and still state the wrong percentage. Calibration cut the Brier score from 0.212 to 0.123.

**Why does the clinician screen show the logistic regression at 77% when the app says 39%?**
The clinician table shows each model's *raw* output; class-weighted models over-state probabilities. The headline number is the calibrated one.

**What would you do to improve it?**
Add information the dataset lacks (family history, lipid fractions, biomarkers; Part 15.6), widen the search (including elastic-net), add a second untouched test set, and validate on another population. Repeated cross-validation, paired comparison, subgroup calibration and decision curves are already done (Part 15).

---

## Part 14. Glossary

- **Parameter**: a number the model learns (a weight).
- **Hyperparameter**: a setting we choose (number of trees, penalty strength).
- **Overfitting**: great on training data, worse on new data.
- **Regularisation**: a penalty that stops a model becoming too complicated (L2 shrinks weights).
- **Cross-validation**: repeatedly training on part of the training data and scoring on the rest.
- **Out-of-fold prediction**: a prediction for a patient made by a model that did not see that patient.
- **Epoch**: one full pass through the training data (neural network).
- **Sigmoid**: a function that squashes any number into 0 to 1, turning a score into a probability.
- **Log-loss**: the training error for probabilities; punishes confident wrong answers.
- **Gini impurity**: how mixed a group of patients is; trees split to reduce it.
- **Bootstrap**: resampling the data with replacement to estimate uncertainty.
- **Calibration**: making predicted percentages match real frequencies.
- **Sensitivity / specificity**: share of sick correctly flagged / share of healthy correctly cleared.
- **Leakage**: information from the test data influencing training.

- **One-standard-error rule**: among settings whose cross-validated score is within one standard error of the best, pick the simplest. It guards against choosing a complicated model because of a lucky fold.
- **Standard error**: how much an average would wobble if you repeated the experiment; here, the spread of the five fold scores divided by the square root of five.

---

## Part 15. The improvement ladder: what we tried to beat Framingham, and what happened

This part exists because the obvious question is "why not just improve the model?" We tried, in a way that does not misuse the test set.

### 15.1 Why we needed a new way to judge improvements
The test set has 848 people and 129 events, so a test AUROC wobbles by about 0.025 just from which people landed in it. Real improvements here are worth 0.005 to 0.01, so the test set cannot see them, and scoring it after every idea would turn it into a tuning set. So every idea was judged on the **3,392 training rows only**, with:

- **5 x 5 repeated cross-validation** (five repeats of five folds, so 25 held-out folds),
- **nested tuning**: the penalty C is chosen inside every training fold by the one-standard-error rule, so the score is not flattered by tuning,
- **preprocessing fitted inside every fold**,
- **paired bootstrap**: each idea and Framingham are scored on the *same* patients and the difference in AUROC is bootstrapped (1,000 resamples). This gives an interval for the difference itself, which is much tighter than comparing two separate intervals.

### 15.2 The steps (each adds one idea to the one before)
| Step | What it adds | Idea |
|---|---|---|
| S2 | Logistic regression on the 23 original features | The model we had |
| S3 | + 8 "Framingham-structure" features: ln(age), ln(SBP), ln(BMI), male x ln(age), male x ln(SBP), male x smoker, male x diabetes, treated-BP x ln(SBP) | Copy the *form* of the clinical equation (logs, separate effects for men and women, a different pressure term when treated) |
| S4 | + "value was missing" flags for glucose, education, BP medication, cholesterol | Perhaps missingness carries information |
| S5b | + the Framingham log-odds as one more feature | Use the clinical score as a starting point; the model learns how much to trust it |
| S5a | Framingham log-odds as an **offset** with weight fixed at 1 | Model *updating*: learn only corrections to the score (`offset_logreg.py`) |
| Extra | Shallow boosting with and without the Framingham feature | Does a flexible model help? |

### 15.3 Results (development data, AUROC)
| Model | AUROC | Difference vs Framingham (95% interval) | Difference vs S2 |
|---|---|---|---|
| Framingham, as published | 0.721 | | |
| Framingham, recalibrated | 0.719 | -0.002 (-0.007 to +0.002) | |
| S2 original 23 features | 0.729 | +0.009 (-0.004 to +0.021) | |
| **S3 + Framingham structure (served)** | **0.734** | **+0.013 (+0.003 to +0.023)** | **+0.004 (+0.001 to +0.009)** |
| S4 + missing flags | 0.734 | +0.013 | +0.005 (nothing beyond S3) |
| S5b + Framingham as a feature | 0.734 | +0.014 | +0.005 (nothing beyond S3) |
| S5a Framingham as offset | 0.727 | +0.006 (-0.001 to +0.012) | -0.003 (worse) |
| Boosting, 23 features | 0.723 | +0.004 (-0.009 to +0.016) | |
| Boosting + Framingham feature | 0.724 | +0.005 (-0.004 to +0.015) | |

What this says in plain words:
- **Only one idea helped: copying the structure of the Framingham equation (S3).** The reason is intuitive: the clinical score is not a straight line in age and pressure; it uses logs and separate effects for men and women, and our plain linear model could not express that. Now it can.
- Missing flags added nothing (the data have little information in missingness).
- Using the score as a feature added nothing on top of S3, because S3 already carries what the score contributes.
- Fixing the score's weight at 1 (offset) was **worse**: it forces the model to keep the score's weaknesses, including its under-prediction in this cohort.
- Boosting did not help.

**Selection rule and an honest disclosure (viva).** We wrote the rule before seeing the results: take the first step whose average precision is within one standard error of the best. That rule picked S2 (the old model), because S3's average precision (0.356) is only 0.007 above S2's (0.349) and one standard error is 0.018. But the paired AUROC comparison showed S3 ahead of S2 with an interval that excludes zero. We then *amended* the rule to: adopt the simplest step whose paired improvement has an interval excluding zero, which gives S3. That change was made after seeing results and we say so. The gain is +0.004 AUROC, the interval is not adjusted for the several steps tried, and on the locked test split S3 does not beat Framingham (0.695 vs 0.703; paired -0.006, interval -0.024 to +0.013). So treat S3's edge as suggestive.

### 15.4 Why scores stop near 0.73 (`ceiling.py`)
- **Ablation** (what each group of inputs buys, development AUROC): age alone 0.690; + sex 0.701; + systolic BP 0.724; + smoking and diabetes 0.732; all 15 raw inputs 0.731; all 31 features 0.731. Five inputs get everything; the other 26 columns add nothing.
- **Learning curve**: held-out AUROC 0.707 with 271 patients, 0.726 with 1,356, 0.731 with 2,034, 0.732 with 2,713, while training AUROC falls from 0.768 to 0.743 and the two meet. More patients of the same kind would not help.
- Conclusion: the ceiling is set by the *information in the inputs*, not by the model. A better learner cannot get past it; new information could.

### 15.5 Is it trustworthy for everyone, and is it useful? (`subgroups.py`)
Out-of-fold predictions on the development data, calibrated by cross-fitting:

| Group | Observed rate | Mean predicted: CardioSense / Framingham as published | Calibration slope: CardioSense / Framingham |
|---|---|---|---|
| Everyone | 15.2% | 15.2% / 12.5% | 0.83 / 0.88 |
| Women | 12.8% | 13.6% / 9.1% | 0.72 / 0.83 |
| Men | 18.3% | 17.3% / 16.9% | 0.97 / 1.11 |
| Age under 50 | 8.0% | 7.8% / 7.2% | 0.75 / 0.86 |
| Age 50 to 59 | 20.0% | 19.1% / 15.2% | 0.90 / 0.85 |
| Age 60 and over | 28.5% | 30.6% / 23.6% | 0.47 / 0.50 |

- The published Framingham score **under-predicts** here (about 20% too low overall, about 30% too low in women). Our model, fitted to this cohort, does not. This is the clearest practical advantage we have.
- A **calibration slope** of 1 is ideal; below 1 means predictions are too spread out (top risks too high, bottom too low). Women (0.72) and people over 60 (0.47) are the weak spots. Say so openly.
- **Within** an age band the AUROC is only 0.61 to 0.67, because most of the overall ranking comes from age itself.
- **Net benefit (decision curve)**: net benefit = TP/N - FP/N x t/(1 - t) at threshold t. At a 15% threshold CardioSense scores 0.053 (about 5.3 true cases found per 100 people with no net cost in false alarms) against 0.047 for published Framingham and 0.002 for flagging everyone. It is level with the *recalibrated* Framingham score, which shows that much of the published score's disadvantage is calibration, not ranking.

### 15.6 Where to get the missing information (HDL/LDL, family history, follow-up time)
- **BioLINCC Framingham teaching dataset** (`frmgham2.csv`, 4,434 participants, 11,627 examinations): has HDL and LDL cholesterol in one examination period and time-to-event variables. It does **not** include family history, and it has to be requested from BioLINCC (NHLBI). Its terms say it is for teaching, not for publication.
- **NHANES with the CDC linked-mortality files**: free to download, includes HDL and LDL, but the outcome is **death**, not coronary heart disease, so it would be a different target.
- Neither could be obtained and validated in the time left, so both are listed as future work in the report. We did **not** invent or simulate these columns.

### 15.7 Viva questions on this part
**You said it beats Framingham in cross-validation but not on the test set. Which is right?** Both are within sampling error. The development comparison has 3,392 people and 515 events and a paired design, so it is more precise, but the test set is the one untouched check. The honest summary is a tie to a small edge.

**Why did you change the selection rule?** The original rule looked only at average precision; the paired AUROC comparison, which is more sensitive, showed a gain. We disclose that the change was made after seeing results.

**Why not use the Framingham score as the base and just correct it (the offset)?** We tried: it was worse than S3 (0.727 vs 0.734) because it keeps the score's under-prediction and structure fixed.

**Why does nothing get past 0.73?** The ablation shows five inputs already reach 0.732 and the learning curve is flat; the information is the limit.
