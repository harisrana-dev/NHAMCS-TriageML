# Random Forest and AdaBoost: nested hospital-grouped CV

## Protocol and scope

This experiment uses the shared preprocessing pipeline and nested runner from
the Part 3 protocol. The outer split is five-fold `StratifiedGroupKFold` on
`HOSPCODE` (`shuffle=True`, seed 42); each outer-training partition has its own
three-fold hospital-grouped inner CV. Macro-F1 over the fixed labels 1–5, with
`zero_division=0`, selects candidates. Preprocessing is refitted inside every
inner training fold and the winning candidate is refitted on the complete
outer-training partition. Each outer test fold is scored once. The parallelism
setting is 1.

Both required models were run in the `full` condition. Before examining the
results, the family-winner rule was fixed as: select the model with the highest
mean of its five `best_inner_score` values; resolve ties by the stable model
order Random Forest, then AdaBoost. Run `no_rfv` only for that selected model.
This rule uses inner-CV scores only, not outer-fold metrics. Random Forest won
with a mean best inner macro-F1 of 0.335383, versus 0.250511 for AdaBoost, so
the only `no_rfv` run is Random Forest.

The searches compared:

| Model | Parameters/strategies evaluated within inner CV |
|---|---|
| Random Forest | `n_estimators={100}`, `max_depth={None,20}`, `max_features={sqrt}`, `min_samples_leaf={1,5}`, `class_weight={None,balanced}` (8 candidates) |
| AdaBoost | `n_estimators={50,100}`, `learning_rate={0.5,1.0}` (4 parameter combinations), each with no weights and inverse-frequency `sample_weight` (8 candidates total) |

AdaBoost does not provide `class_weight`; inverse-frequency weights
`w_c = n / (5 * n_c)` are computed from each fit partition only. Inner-CV
selection chooses the weighting strategy alongside the AdaBoost parameters;
the chosen strategy is reapplied using the outer-training labels for the final
fold fit. Ties follow parameter-grid order and then the declared strategy
order, which puts unweighted before weighted.

## Reproducibility

The run used the verified NHAMCS 2022 file (16,025 rows, 913 columns; SHA-256
`0E00CD5AEA4511CD619F7AF1E0DC2F6825F111B7C87D3EBBB1358410A4C881C2`) and the
10,207-visit cohort (147 hospitals; labels 1–5). Environment: Windows,
CPython 3.12.10, NumPy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1. The seed is
42. The dataset digest, library versions, split settings, candidate counts,
and fit counts are recorded in `reports/ensemble_nested/metadata.json`.

From the repository root with the dataset at `data/local/ed2022-stata.dta`:

```powershell
$env:NHAMCS_DATA_PATH = (Resolve-Path .\data\local\ed2022-stata.dta).Path
python -m scripts.run_ensemble_nested
```

The committed CSVs under `reports/ensemble_nested/` contain fold metrics,
pooled OOF metrics and predictions, per-class metrics, per-fold and pooled
confusion matrices, inner-search scores, selected parameters/weighting policy
per outer fold, and outer/inner split diagnostics.

## Results

Mean and sample standard deviation (`ddof=1`) across the five outer folds:

| Model | Condition | Macro-F1 | Accuracy | Balanced accuracy | Macro ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | full | **0.3335 ± 0.0115** | 0.5048 ± 0.0286 | 0.3443 ± 0.0249 | 0.7177 ± 0.0092 |
| AdaBoost | full | 0.2390 ± 0.0124 | 0.3098 ± 0.0362 | **0.3629 ± 0.0177** | 0.6622 ± 0.0179 |
| Random Forest | no_rfv | 0.2778 ± 0.0130 | 0.4301 ± 0.0127 | 0.2775 ± 0.0078 | 0.6260 ± 0.0142 |

Pooled OOF macro-F1 / accuracy / balanced accuracy:

| Model | Condition | Macro-F1 | Accuracy | Balanced accuracy |
|---|---|---:|---:|---:|
| Random Forest | full | 0.3384 | 0.5048 | 0.3432 |
| AdaBoost | full | 0.2476 | 0.3099 | 0.3625 |
| Random Forest | no_rfv | 0.2795 | 0.4302 | 0.2778 |

Selected parameters by outer fold (all five AdaBoost folds selected
inverse-frequency weighting; all Random Forest folds selected
`class_weight="balanced"`):

| Model/condition | Fold | Best inner macro-F1 | Selected configuration |
|---|---:|---:|---|
| Random Forest / full | 0 | 0.3173 | balanced, depth 20, sqrt, leaf 1, 100 trees |
| Random Forest / full | 1 | 0.3450 | balanced, unlimited depth, sqrt, leaf 1, 100 trees |
| Random Forest / full | 2 | 0.3440 | balanced, unlimited depth, sqrt, leaf 1, 100 trees |
| Random Forest / full | 3 | 0.3380 | balanced, unlimited depth, sqrt, leaf 1, 100 trees |
| Random Forest / full | 4 | 0.3325 | balanced, unlimited depth, sqrt, leaf 1, 100 trees |
| AdaBoost / full | 0 | 0.2422 | inverse-frequency, learning rate 1.0, 100 estimators |
| AdaBoost / full | 1 | 0.2524 | inverse-frequency, learning rate 1.0, 100 estimators |
| AdaBoost / full | 2 | 0.2688 | inverse-frequency, learning rate 1.0, 100 estimators |
| AdaBoost / full | 3 | 0.2509 | inverse-frequency, learning rate 1.0, 100 estimators |
| AdaBoost / full | 4 | 0.2383 | inverse-frequency, learning rate 0.5, 100 estimators |
| Random Forest / no_rfv | 0 | 0.2616 | balanced, unlimited depth, sqrt, leaf 1, 100 trees |
| Random Forest / no_rfv | 1 | 0.2788 | balanced, unlimited depth, sqrt, leaf 1, 100 trees |
| Random Forest / no_rfv | 2 | 0.2769 | balanced, depth 20, sqrt, leaf 1, 100 trees |
| Random Forest / no_rfv | 3 | 0.2790 | balanced, unlimited depth, sqrt, leaf 1, 100 trees |
| Random Forest / no_rfv | 4 | 0.2767 | balanced, depth 20, sqrt, leaf 1, 100 trees |

The full Random Forest condition has the highest macro-F1 and ROC-AUC in this
run; AdaBoost has the highest mean balanced accuracy. Both show a substantial
gap between the rare classes and the more frequent middle classes. Per-class
scores are fold-specific in `per_class_metrics.csv`; pooled class-level
precision, recall, and F1 are also exported in `pooled_per_class_metrics.csv`.

Pooled OOF confusion matrices (rows are true labels, columns are predictions):

**Random Forest, full**

| True \ Predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 11 | 72 | 38 | 18 | 0 |
| 2 | 12 | 713 | 553 | 299 | 18 |
| 3 | 18 | 999 | 2649 | 1619 | 55 |
| 4 | 6 | 206 | 787 | 1758 | 69 |
| 5 | 1 | 19 | 73 | 192 | 22 |

**AdaBoost, full**

| True \ Predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 71 | 23 | 11 | 23 | 11 |
| 2 | 418 | 422 | 228 | 360 | 167 |
| 3 | 896 | 619 | 1312 | 1651 | 862 |
| 4 | 224 | 78 | 515 | 1251 | 758 |
| 5 | 35 | 5 | 32 | 128 | 107 |

**Random Forest, no_rfv**

| True \ Predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 14 | 63 | 43 | 18 | 1 |
| 2 | 20 | 529 | 697 | 333 | 16 |
| 3 | 22 | 986 | 2570 | 1679 | 83 |
| 4 | 6 | 267 | 1236 | 1270 | 47 |
| 5 | 0 | 34 | 123 | 142 | 8 |

All five classes occur in every outer test fold. The fold diagnostics show 0
hospital overlap, with 26–31 held-out hospitals per fold. Rare class 1 has
24–34 test visits per fold and class 5 has 61–63; their fold-level estimates
are therefore noisy. Only one seed and one five-fold outer partition were
used, as specified by the protocol.

## Reproduction cross-check

Before preparing this change for review, the GaussianNB Part 2 experiment was
rerun through the shared `run_evaluation` pipeline on the same verified file
and environment. The reproduced fold-mean macro-F1 / accuracy were 0.158519 /
0.162637 (`full`) and 0.098656 / 0.115061 (`no_rfv`), matching the rounded
values in `docs/gaussian_nb_experiment.md`.
