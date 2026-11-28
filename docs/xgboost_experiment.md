# XGBoost Experiment — Issue #13

## 1. Scope

This report documents the XGBoost experiment using the project's shared
preprocessing and nested hospital-grouped cross-validation evaluator.

Only the `full` feature condition was evaluated.

The `no_rfv` condition was not run because it is reserved for a model
selected as a pre-declared family winner.

## 2. Dataset

| Item | Value |
|---|---|
| Model | XGBoost |
| Dataset rows | 10,207 |
| Hospitals | 147 |
| Target | IMMEDR |
| Classes | 1–5 |
| Feature condition | full |
| Outer folds | 5 |
| Inner folds | 3 |
| Random state | 42 |
| Group column | HOSPCODE |

## 3. Model configuration

| Parameter | Value |
|---|---|
| XGBoost version | 3.4.1 |
| Objective | multi:softprob |
| Evaluation metric | mlogloss |
| Tree method | hist |
| Number of estimators | 100 |
| Learning rate | 0.05 |
| Tested max depths | 3, 5 |
| Estimator parallelism | 1 |
| Search parallelism | 1 |
| Weighting candidates | None, inverse frequency |

The inner cross-validation selected the candidate using macro-F1.
Sample weights were calculated from each training partition, avoiding
use of validation or outer test labels during weight calculation.

## 4. Outer-fold results

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Macro-F1 | 0.325958 | 0.013313 |
| Accuracy | 0.433763 | 0.015513 |
| Balanced accuracy | 0.404428 | 0.027563 |
| ROC-AUC | 0.711882 | 0.004600 |

All five outer folds selected:
- Maximum depth: 5
- Weight policy: inverse_frequency

## 5. Pooled out-of-fold results

All 10,207 held-out predictions were combined to calculate these metrics.

| Metric | Result |
|---|---:|
| Macro-F1 | 0.326150 |
| Accuracy | 0.433820 |
| Balanced accuracy | 0.403551 |

### Per-class performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| 1 | 0.0627 | 0.2806 | 0.1025 |
| 2 | 0.3667 | 0.4690 | 0.4116 |
| 3 | 0.7109 | 0.3491 | 0.4682 |
| 4 | 0.4317 | 0.5934 | 0.4998 |
| 5 | 0.0963 | 0.3257 | 0.1487 |

Classes 1 and 5 have considerably lower F1 scores than classes 2, 3,
and 4. This shows that performance varies across the five classes.

## 6. Reproducibility
Install the optional XGBoost dependency using `python -m pip install -r requirements-xgboost.txt`.
Run the experiment from the repository root using `python -m scripts.run_xgboost`.
Results are saved in `reports/xgboost_full/`, including fold metrics, pooled out-of-fold metrics, per-class metrics, confusion matrix, predictions, best parameters, inner search results, diagnostics, and metadata.
## 7. Validation and remaining work
Python compilation passed. The weighting tests passed (6 tests), and the full project test suite passed (70 tests).
Before opening a pull request, reproduce at least one other team member's result and document the comparison. This requirement is still outstanding. The no_rfv condition has not been evaluated.
