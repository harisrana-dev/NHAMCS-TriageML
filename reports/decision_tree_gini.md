# Gini Decision Tree experiment

## Reproduction

Run from the repository root with the verified NHAMCS 2022 Stata file available
at the path in `NHAMCS_DATA_PATH`. The raw file is not committed.

Environment used: CPython 3.12.10, pandas 2.3.3, NumPy 2.5.4,
scikit-learn 1.9.1, SciPy 1.18.1, pytest 9.1.1.

```powershell
$env:NHAMCS_DATA_PATH = (Join-Path (Get-Location) 'ed2022-stata.dta')
python -m pytest -q
@'
from src.data.dataset import load_raw_dataset
from src.data.target import select_modelling_population
from src.evaluation.data import prepare_modelling_inputs
from src.evaluation.runner import run_evaluation

inputs = prepare_modelling_inputs(
    select_modelling_population(load_raw_dataset())
)
result = run_evaluation(inputs, model_ids=["decision_tree_gini"])
print(result.summary.to_string(index=False))
print(result.fold_metrics.to_string(index=False))
print(result.per_class_metrics.to_string(index=False))
print(result.summary_for("decision_tree_gini", "full"))
print(result.summary_for("decision_tree_gini", "no_rfv"))
print(result.confusion_matrices[("decision_tree_gini", "full")])
print(result.confusion_matrices[("decision_tree_gini", "no_rfv")])
print(result.pooled_oof_metrics.to_string(index=False))
'@ | python -
```

The experiment used the existing evaluator and factory:
`DecisionTreeClassifier(criterion="gini", random_state=42)`. Its default
conditions are `full` (20 features) and `no_rfv` (15 features); both use the
same five hospital-grouped folds. The evaluator fits preprocessing and the
estimator within each training fold.

The local input was 37,720,094 bytes and matched SHA-256
`0E00CD5AEA4511CD619F7AF1E0DC2F6825F111B7C87D3EBBB1358410A4C881C2`.
It loaded as 16,025 rows × 913 columns. The modelling cohort had 10,207 visits
from 147 hospitals, with `IMMEDR` counts 1: 139, 2: 1,595, 3: 5,340, 4: 2,826,
and 5: 307.

Validation on this branch: `python -m pytest -q` — **55 passed**. With
`NHAMCS_DATA_PATH` set to the repository-root dataset,
`python -m pytest -m integration -q` — **3 passed, 52 deselected**.

## Cross-validation summary

Fold standard deviations are sample standard deviations (`ddof=1`). ROC-AUC was
available for all five folds in each condition.

| Condition | Macro-F1 mean ± SD | Accuracy mean ± SD | Balanced accuracy mean ± SD | ROC-AUC mean ± SD |
|---|---:|---:|---:|---:|
| full | 0.2799 ± 0.0198 | 0.4627 ± 0.0075 | 0.2774 ± 0.0195 | 0.5509 ± 0.0104 |
| no_rfv | 0.2456 ± 0.0151 | 0.4013 ± 0.0095 | 0.2452 ± 0.0138 | 0.5269 ± 0.0087 |

### Per-fold results

| Condition | Fold | Macro-F1 | Accuracy | Balanced accuracy | ROC-AUC | ROC-AUC status |
|---|---:|---:|---:|---:|---:|---|
| full | 0 | 0.3035 | 0.4686 | 0.3036 | 0.5646 | available |
| full | 1 | 0.2676 | 0.4625 | 0.2669 | 0.5462 | available |
| full | 2 | 0.2529 | 0.4510 | 0.2517 | 0.5365 | available |
| full | 3 | 0.2867 | 0.4617 | 0.2814 | 0.5531 | available |
| full | 4 | 0.2889 | 0.4697 | 0.2835 | 0.5541 | available |
| no_rfv | 0 | 0.2315 | 0.3873 | 0.2340 | 0.5184 | available |
| no_rfv | 1 | 0.2421 | 0.4061 | 0.2441 | 0.5283 | available |
| no_rfv | 2 | 0.2714 | 0.4131 | 0.2690 | 0.5412 | available |
| no_rfv | 3 | 0.2417 | 0.3997 | 0.2377 | 0.5226 | available |
| no_rfv | 4 | 0.2415 | 0.4004 | 0.2413 | 0.5242 | available |

The evaluator reported no unavailable ROC-AUC folds; the reason field is empty
for each.

Per-fold precision, recall, and F1 for each of classes 1–5 are reported in
`docs/decision_tree_fold_metrics.csv`, generated from
`result.fold_metrics`/`result.per_class_metrics` by
`python -m scripts.fold_metrics`. ROC-AUC availability and reason are included
for each fold.

## Pooled out-of-fold results

These metrics pool the held-out predictions for all 10,207 visits; they are not
averages of the fold scores.

| Condition | Macro-F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| full | 0.2800 | 0.4627 | 0.2778 |
| no_rfv | 0.2450 | 0.4013 | 0.2447 |

### Pooled per-class precision, recall and F1

| Condition | `IMMEDR` | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| full | 1 | 0.0932 | 0.0791 | 0.0856 |
| full | 2 | 0.2960 | 0.2771 | 0.2863 |
| full | 3 | 0.5667 | 0.5865 | 0.5764 |
| full | 4 | 0.3998 | 0.3974 | 0.3986 |
| full | 5 | 0.0577 | 0.0489 | 0.0529 |
| no_rfv | 1 | 0.1163 | 0.1079 | 0.1119 |
| no_rfv | 2 | 0.2099 | 0.2157 | 0.2127 |
| no_rfv | 3 | 0.5334 | 0.5170 | 0.5251 |
| no_rfv | 4 | 0.3290 | 0.3408 | 0.3348 |
| no_rfv | 5 | 0.0387 | 0.0423 | 0.0404 |

### Pooled out-of-fold confusion matrices

Rows are true `IMMEDR`; columns are predicted `IMMEDR`, in class order 1–5.

**full**

| True \ Predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 11 | 38 | 68 | 21 | 1 |
| 2 | 39 | 442 | 826 | 264 | 24 |
| 3 | 51 | 755 | 3,132 | 1,278 | 124 |
| 4 | 17 | 224 | 1,366 | 1,123 | 96 |
| 5 | 0 | 34 | 135 | 123 | 15 |

**no_rfv**

| True \ Predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 15 | 41 | 59 | 21 | 3 |
| 2 | 35 | 344 | 793 | 382 | 41 |
| 3 | 67 | 852 | 2,761 | 1,471 | 189 |
| 4 | 12 | 352 | 1,409 | 963 | 90 |
| 5 | 0 | 50 | 154 | 90 | 13 |

## Interpretation and limits

- On these fixed folds, `full` scored higher than `no_rfv` on the reported
  aggregate metrics. The Macro-F1 mean difference is 0.0343. This is an
  observation about these conditions and folds, not evidence that RFV features
  must help on other data or a causal effect of RFV.
- `IMMEDR` 1 is 139/10,207 (1.36%) of this cohort. Its pooled recall is 0.0791
  with `full` and 0.1079 with `no_rfv`; the rare class remains difficult for
  this classifier. Class 5 also has low pooled F1 in both conditions. These are
  performance observations, not clinical conclusions.
- Fold Macro-F1 ranges from 0.2529 to 0.3035 (`full`) and 0.2315 to 0.2714
  (`no_rfv`). The observed standard deviations describe fold variability. The
  evaluator does not retain train-set scores, so these results alone do not
  establish overfitting.
- A decision tree can be represented as a set of sequential feature tests and
  terminal class decisions, making its decision mechanism more directly
  expressible as rules than GaussianNB's class-conditional probability model;
  the Dummy baseline has no learned feature rules. For the observed fold-level
  rules and feature importances, `scripts.tree_splits_all_folds` reads the exact
  fitted pipelines retained by the shared evaluator; it does not regenerate
  folds or refit trees. The trees are large (depth 64–87 in `full` and 36–45 in
  `no_rfv`), so only their upper split levels are readily interpretable.
