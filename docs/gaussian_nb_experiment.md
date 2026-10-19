# GaussianNB experiment — Issue #2 write-up

**Issue:** [Part 2: Add GaussianNB experiment and document assumptions](https://github.com/harisrana-dev/NHAMCS-TriageML/issues/2)
**Branch:** `feat/fiza/issue-2-gaussian-nb`
**Status:** executed on the pinned dataset through the shared Part 2 evaluator. Every metric below was produced by `src.evaluation.runner.run_evaluation` on 2026-10-19. Counts and percentages derived from those evaluator outputs (for example `268 / 5,340 = 0.0502`, the 5,072 remainder, 3,077 class-1 predictions, and the class shares in the table below) were calculated manually from the reported tables and independently checked; nothing is estimated or carried over from another machine.

## Scope

This document reports the **Gaussian Naive Bayes** experiment for Issue #2. No
evaluation logic was reimplemented: the run goes through the shared Issue #1
evaluator exactly as prescribed — one shared list of hospital-grouped folds,
fold-local preprocessing, both approved feature conditions, no tuning.

```python
from src.data.dataset import load_raw_dataset
from src.data.target import select_modelling_population
from src.evaluation.data import prepare_modelling_inputs
from src.evaluation.runner import run_evaluation

inputs = prepare_modelling_inputs(select_modelling_population(load_raw_dataset()))
result = run_evaluation(inputs, model_ids=["gaussian_nb"])
```

## Environment and dataset

| Item | Value |
|---|---|
| Python | 3.14.2 |
| pandas | 2.3.3 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| Dataset | `data/local/ed2022-stata.dta` (official CDC/NCHS download) |
| Integrity | SHA-256 and size verified by `load_raw_dataset(verify=True)`; shape `(16025, 913)` confirmed |
| Modelling cohort | 10,207 rows, 147 hospitals, `IMMEDR` ∈ {1,2,3,4,5} |
| Class shares | 1: 139 (1.36%), 2: 1,595 (15.63%), 3: 5,340 (52.32%), 4: 2,826 (27.69%), 5: 307 (3.01%) |
| Folds | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, groups=`HOSPCODE`; generated once and reused |
| Estimator | `GaussianNB()` (defaults, no tuning), fresh instance per fold via `MODEL_FACTORIES["gaussian_nb"]` |
| Conditions | `full` = 20 approved columns; `no_rfv` = the same register minus the 5 RFV columns = 15 columns |

Fold hygiene (from `result.diagnostics`): hospital overlap between train and
test is **0** in every fold; test folds hold out 26–31 hospitals and 2,031–2,064
rows; **all five classes are present in every test fold**, so the fixed
five-class macro ROC-AUC is defined for all 5 folds in both conditions
(`roc_auc_n_valid = 5`).

## Results

### Summary (`result.summary` / `result.summary_for`)

Fold mean ± sample standard deviation (`ddof=1`), 5 folds each:

| Model | Condition | Macro-F1 | Accuracy | Balanced acc. | ROC-AUC (OvR macro) |
|---|---|---|---|---|---|
| gaussian_nb | full | **0.1585 ± 0.0069** | 0.1626 ± 0.0074 | 0.3209 ± 0.0168 | 0.5699 ± 0.0142 |
| gaussian_nb | no_rfv | **0.0987 ± 0.0507** | 0.1151 ± 0.0731 | 0.2864 ± 0.0195 | 0.5960 ± 0.0130 |

Reference point from Part 1 documentation (not a model output): the
majority-class accuracy floor is 0.5232 (`IMMEDR = 3`). GaussianNB trades
accuracy away from the majority class; both conditions sit far below that floor
on accuracy while predicting all five classes.

### Per-fold metrics (`result.fold_metrics`)

| Condition | Fold | n_test | Macro-F1 | Accuracy | Balanced acc. | ROC-AUC | AUC available |
|---|---|---|---|---|---|---|---|
| full | 0 | 2040 | 0.1617 | 0.1681 | 0.3009 | 0.5558 | yes |
| full | 1 | 2039 | 0.1544 | 0.1555 | 0.3227 | 0.5704 | yes |
| full | 2 | 2031 | 0.1499 | 0.1561 | 0.3069 | 0.5554 | yes |
| full | 3 | 2064 | 0.1588 | 0.1613 | 0.3399 | 0.5831 | yes |
| full | 4 | 2033 | 0.1678 | 0.1722 | 0.3338 | 0.5846 | yes |
| no_rfv | 0 | 2040 | 0.0543 | 0.0446 | 0.2646 | 0.5793 | yes |
| no_rfv | 1 | 2039 | 0.1197 | 0.1344 | 0.2887 | 0.6070 | yes |
| no_rfv | 2 | 2031 | 0.1256 | 0.1448 | 0.2742 | 0.6069 | yes |
| no_rfv | 3 | 2064 | 0.1566 | 0.2122 | 0.3163 | 0.6021 | yes |
| no_rfv | 4 | 2033 | 0.0371 | 0.0394 | 0.2883 | 0.5849 | yes |

Every fold's `roc_auc_reason` is empty (AUC computed; no unavailability). A
single fold is never the reported result — the tables above and the summary in
the previous section are.

### Per-class metrics (`result.per_class_metrics`, mean ± std over folds)

| Condition | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| full | 1 Immediate | 0.0224 ± 0.0059 | 0.4910 ± 0.0714 | 0.0428 ± 0.0109 |
| full | 2 Emergent | 0.2315 ± 0.0224 | 0.3731 ± 0.0267 | 0.2855 ± 0.0239 |
| full | 3 Urgent | 0.6238 ± 0.0353 | 0.0502 ± 0.0087 | 0.0927 ± 0.0149 |
| full | 4 Semi-urgent | 0.4078 ± 0.0449 | 0.2049 ± 0.0042 | 0.2722 ± 0.0106 |
| full | 5 Non-urgent | 0.0554 ± 0.0058 | 0.4852 ± 0.0526 | 0.0994 ± 0.0101 |
| no_rfv | 1 Immediate | 0.0336 ± 0.0148 | 0.4663 ± 0.1221 | 0.0618 ± 0.0259 |
| no_rfv | 2 Emergent | 0.2787 ± 0.0667 | 0.0853 ± 0.0483 | 0.1235 ± 0.0602 |
| no_rfv | 3 Urgent | 0.3889 ± 0.4157 | 0.0011 ± 0.0015 | 0.0022 ± 0.0031 |
| no_rfv | 4 Semi-urgent | 0.2656 ± 0.1511 | 0.2737 ± 0.2546 | 0.2326 ± 0.1854 |
| no_rfv | 5 Non-urgent | 0.0392 ± 0.0045 | 0.6057 ± 0.1955 | 0.0730 ± 0.0072 |

### Pooled out-of-fold metrics (`result.pooled_oof_metrics`, n = 10,207)

| Condition | Macro-F1 | Accuracy | Balanced acc. |
|---|---|---|---|
| full | 0.1585 | 0.1626 | 0.3220 |
| no_rfv | 0.1106 | 0.1153 | 0.2875 |

Pooled per-class (precision / recall / F1):

| Condition | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| full | 0.0224 / 0.4964 / 0.0429 | 0.2308 / 0.3730 / 0.2852 | 0.6218 / 0.0502 / 0.0929 | 0.4046 / 0.2049 / 0.2720 | 0.0554 / 0.4853 / 0.0994 |
| no_rfv | 0.0277 / 0.4676 / 0.0524 | 0.2720 / 0.0853 / 0.1298 | 0.4000 / 0.0011 / 0.0022 | 0.3192 / 0.2774 / 0.2969 | 0.0380 / 0.6059 / 0.0715 |

### Confusion matrices (`result.confusion_matrices`, pooled OOF; rows = true, columns = predicted)

`("gaussian_nb", "full")`:

| true \ predicted | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 69 | 35 | 5 | 13 | 17 |
| 2 | 611 | 595 | 74 | 114 | 201 |
| 3 | 1662 | 1515 | 268 | 681 | 1214 |
| 4 | 665 | 398 | 75 | 579 | 1109 |
| 5 | 70 | 35 | 9 | 44 | 149 |

`("gaussian_nb", "no_rfv")`:

| true \ predicted | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 65 | 28 | 3 | 13 | 30 |
| 2 | 534 | 136 | 0 | 308 | 617 |
| 3 | 1261 | 257 | 6 | 1286 | 2530 |
| 4 | 431 | 75 | 6 | 784 | 1530 |
| 5 | 52 | 4 | 0 | 65 | 186 |

## Analysis

This section separates **observations** (facts produced by the evaluator
tables above) from **hypotheses** (mechanistic explanations consistent with the
observations, not proven by them). No clinical claim is made or implied: the
model predicts a documentation field, not patient outcomes.

### 1. Conditional-independence assumption, and RFV1/RFV2 by construction

Naive Bayes factorises the class-conditional likelihood as if the features were
independent given the class. The five RFV slots are collected **in the same
visit, from the same complaint list, ranked by the recorder**: `RFV1` is the
primary reason, `RFV2` the secondary, and so on. By construction they are
dependent — a ranked restatement of one presenting complaint, not five
independent observations. Feeding them to NB as if independent multiplies
near-duplicate evidence, over-counting the complaint signal in the likelihood
product.

**Observation:** with the RFV columns included (`full`), the model predicts
class 3 only rarely — 431 of 10,207 out-of-fold predictions — and recall on
class 3 collapses to 0.0502 (pooled): only 268 of 5,340 true class-3 rows are
classified as class 3, with the other 5,072 spread across the remaining
classes. The rare class-3 predictions are made confidently, not tentatively:
the aligned out-of-fold posteriors from the same shared folds and pipelines
show 430 of those 431 at ≥ 0.7 for class 3 (counted manually and checked
against the confusion-matrix column). The problem is how seldom class 3 is
chosen, not how confident the model is when it is chosen. Without RFV
(`no_rfv`), the prediction mass collapses onto classes 4/5 — only 15 of 10,207
predictions are class 3 — and class 3 recall falls further to 0.0011 (6 of
5,340).

**Hypothesis:** the conditional-independence violation weakens NB in both
conditions — with RFV because duplicated complaint evidence dominates the
posterior, without RFV because the remaining 15 features carry too little
class-separating signal for the Gaussian likelihood terms to compensate. That
weakness is a legitimate, explainable finding of the model class, not a defect
of the pipeline.

### 2. Gaussian feature-distribution assumption vs discrete inputs

Of the 20 approved columns, only 7 are continuous measurements (vitals,
arrival time). The rest are discrete: 2 binary method flags (passthrough, 0/1),
6 low-cardinality categoricals and 5 RFV categories, all **one-hot encoded**
inside the fold pipeline. A Gaussian likelihood on a 0/1 one-hot column models a
Bernoulli variable with a bell curve: the per-class "mean" is the conditional
frequency, and the per-class "variance" for near-deterministic columns
bottoms out at the `var_smoothing` floor. `var_smoothing` adds a small
variance floor — scikit-learn sets `epsilon_ = var_smoothing × max var(X)`
(1e-9 × the largest variance among the training-fold features) and adds it to
every per-class variance — purely to stabilise the estimates numerically; it
does not meaningfully broaden any distribution. Because NB multiplies one
likelihood term per column, the many one-hot columns contribute a large number
of these mismatched terms.

**Observation:** hard-label accuracy is 0.1626 (`full`) and 0.1151 (`no_rfv`),
while fold ROC-AUC is 0.5699 and 0.5960 — the ranking quality is modestly above
chance even where argmax decisions are poor.

**Hypothesis:** probability ranking degrades more gracefully than thresholded
decisions under a misspecified likelihood, because class-conditional *ordering*
of visit profiles survives even when the absolute posterior masses (which the
misspecified Gaussian terms distort) do not. This is consistent with the
AUC-vs-accuracy gap but is not demonstrated by it.

### 3. Minority-class behaviour (IMMEDR = 1, 139 rows, 1.36%)

**Observations:**
- `full`: class 1 recall 0.4964 at precision 0.0224 (pooled). The confusion
  matrix shows 3,077 rows predicted as class 1 across all true classes, against
  only 139 true class-1 rows.
- `no_rfv`: similar recall (0.4676) at precision 0.0277; fold-to-fold recall
  varies from 0.3333 to 0.6400.
- The majority class (3, 52.32% of rows) is *neglected*: recall 0.0502
  (`full`), 0.0011 (`no_rfv`).

**Hypothesis:** with only ~110 class-1 rows per training fold, the Gaussian
parameters for class 1 — means and class-specific variances alike — are
estimated from very little data, so those variance estimates are noisy. Noisy
class-specific variances making class-1 likelihood terms broad enough to win
the argmax for many non-class-1 visits is a possible explanation, not a proven
cause; the `var_smoothing` floor itself is far too small (1e-9 × the largest
feature variance) to broaden them on its own. The result is high recall at
near-zero precision — the model flags "immediate" far too liberally. Any
deployment reading of this behaviour would be a decision-threshold question,
not something this experiment resolves; per-issue, no clinical inference is
drawn from it.

### 4. `full` vs `no_rfv` sensitivity (no condition assumed to win)

**Observations:**
- Hard-label metrics favour `full` and are stable there: macro-F1 0.1585 ±
  0.0069 vs 0.0987 ± 0.0507 (`no_rfv`); pooled macro-F1 0.1585 vs 0.1106.
  `no_rfv` is unstable — folds 0 and 4 collapse to macro-F1 0.0543 and 0.0371,
  and in those runs entire classes receive zero recall (class 3 in folds 0 and
  2; class 4 in fold 4).
- Fold ROC-AUC does **not** favour `full`: 0.5699 ± 0.0142 vs 0.5960 ± 0.0130.
  The five paired fold differences (`full` − `no_rfv`) are −0.0235, −0.0366,
  −0.0515, −0.0190 and −0.0003: the `no_rfv` ranking is better in all five
  folds, but fold 4 is a near-tie (0.5846 vs 0.5849), and the advantage holds
  even in the folds where its argmax predictions collapse.
- `no_rfv` prediction behaviour swings widely across folds (macro-F1 range
  0.0371–0.1566) while `full` stays in a narrow band (0.1499–0.1678).

**Hypotheses (distinguished from the observations above):** the RFV columns
appear to supply most of the *decision-level* signal that keeps NB's argmax
stable across hospital groups, while their independent-counting inflation may
slightly *hurt* probability ranking — a possible mechanism for the reversed
ROC-AUC ordering. The mean AUC gap is 0.026, and the five paired fold
differences range from 0.0003 to 0.0515 in `no_rfv`'s favour. No statistical
test was run on them, so no significance is claimed in either direction; the
gap should be read as a sensitivity signal, not a conclusion. Neither
condition is assumed to be the "correct" one; the comparison shows the RFV
assumption documented in Part 1 is consequential for this model class in both
directions.

### 5. Reproducibility notes for Issue #5 integration

- Folds are shared across models and conditions (`random_state=42`,
  `HOSPCODE` groups), so these numbers are directly comparable with the other
  Part 2 model experiments run through the same evaluator.
- The evaluator returns per-fold metrics, fold-mean ± sample std (`ddof=1`) and
  pooled OOF together — it does not designate any single fold as *the* result.
  Every headline figure above is the fold mean ± sample std (`ddof=1`) and/or
  pooled OOF over all 10,207 rows.
- Runtime of `run_evaluation(inputs, model_ids=["gaussian_nb"])` on the
  recorded environment: ~3 s after data load; data load + verification ~10 s.

## Validation

Run from the repository root with the pinned dataset in place:

```bash
# Data smoke test (resolve + integrity + load)
python -c "from src.data.dataset import load_raw_dataset; from src.config import resolve_dataset_path; print(resolve_dataset_path()); print(load_raw_dataset().shape)"
# -> data/local/ed2022-stata.dta ; (16025, 913)

python -m pytest -q
# -> 49 passed in 17.84s

python -m pytest -m integration -q
# -> 3 passed, 46 deselected in 12.48s
```

Both commands were executed on 2026-10-19 in the environment recorded above;
the integration tests resolved the real pinned dataset (they were not skipped).

## Constraints honoured

- GaussianNB only through `run_evaluation(inputs, model_ids=["gaussian_nb"])`;
  no independent CV loop, no second preprocessing stack.
- No preprocessing fitted on the full dataset before CV — every fold fits a
  fresh `Pipeline` on training hospitals only.
- No hyperparameter tuning or search; `GaussianNB()` defaults.
- `HOSPCODE` (and `PATCODE`/`CPSUM`/`CSTRATM`) never enter `X`; enforced by
  `leakage_checks` at pipeline construction.
- No raw dataset, patient-level records, or patient-level prediction dumps are
  committed.
