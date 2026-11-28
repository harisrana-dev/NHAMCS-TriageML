# Regularized logistic regression for triage

[Issue #14](https://github.com/harisrana-dev/NHAMCS-TriageML/issues/14)
implements the assignment's three regularization families for the five
`IMMEDR` triage classes. Multinomial logistic regression estimates a
probability for each class and regularizes its coefficients. Ridge applies
L2, Lasso applies L1, and Elastic Net combines the two. Ordinary Ridge, Lasso,
and ElasticNet regressors optimize a continuous-target objective, so the
experiment uses the multinomial classifier for all three families.

## Protocol

All three families use the shared `full` feature register and preprocessing
pipeline, including its numeric `StandardScaler`. The pipeline is fitted
afresh on each inner training partition and again on each complete outer
training partition. `HOSPCODE` is used only to group splits. The shared
evaluation harness uses five outer and three inner shuffled
`StratifiedGroupKFold` folds with seed 42. The same outer folds apply to all
three families, and the inner folds are recreated deterministically from each
outer training partition. Both levels check for hospital overlap.

Each family has its own inner search. Candidate selection maximizes mean
inner macro-F1 over labels 1–5 with `zero_division=0`. Ties go to the first
maximum in scikit-learn's `ParameterGrid` order.

All families search `C` = 0.01, 0.1, 1, 10, 100. Smaller values apply
stronger regularization. All three families use a fixed unweighted fit
(`class_weight=None`). With the pinned scikit-learn 1.9.1 API, SAGA fits the
multinomial objective for these five-class data and `l1_ratio` sets the
penalty. Ridge fixes it at 0, Lasso fixes it at 1, and Elastic Net searches
0.1, 0.5, 0.9. That gives 5, 5, and 15 candidates, respectively.

## Reproduction

From the repository root with the verified NHAMCS 2022 Stata file available
under `data/local/` or via `NHAMCS_DATA_PATH`:

```bash
python -m scripts.run_logistic_nested
python -m pytest -q -p no:cacheprovider
```

The final solver uses seed 42, `max_iter=15000`, and `tol=0.005`. The initial
5,000-iteration budget and `tol=0.0001` produced a convergence warning in
a diagnostic balanced Ridge fit on the first inner training partition. The
adjusted run converts the shared design matrix to CSR for SAGA without
changing feature values. With `tol=0.001`, the balanced Ridge candidate at `C=0.01` still
failed on outer fold 0, inner fold 0 after 15,000 iterations. Since weighting
was optional, the run fixed all three families to unweighted fits before
any held-out fold was scored.

A subsequent runtime diagnostic used only the first inner-training split.
The unweighted Lasso candidate at `C=100` needed 571 iterations at
`tol=0.001` and 106 at `tol=0.005`. The complete experiment was restarted
at the latter tolerance for all three families. The C and mixing-ratio
grids and the selection rule stayed fixed. Outer-fold metrics were not
used for any solver or weighting adjustment.

The run aborts on any `ConvergenceWarning`, including during inner search and
outer refits. Both run in the same process under the sequential #11 protocol.
The solver settings, file SHA-256, package versions, and full search protocol
are recorded in [metadata.json](../reports/logistic_nested/metadata.json).

The [result directory](../reports/logistic_nested/) follows the existing
nested-ensemble export schema for issue #15. It contains fold metrics,
fold means and sample standard deviations, pooled OOF metrics, per-class
scores, confusion matrices, split diagnostics, all inner scores, and the
selected settings and best inner score for each outer fold. The OOF table
uses the harness's `row_index` and exports only the result fields and
probabilities. `proba_1` through `proba_5` follow class order 1 through 5.
Pooled metrics use the concatenated held-out predictions for each family.
Fold summaries describe variability across this single seeded partition.

## Results

The final run completed all 390 fits without a convergence warning.
The exported results were checked against the verified dataset and its
grouped splits. All 10,207 rows have exactly one held-out prediction per
family, and all 15 model-fold pairs have saved selections. The check
confirmed zero hospital overlap and reproduced the pooled metrics from the
exported predictions.

| Family | Fold macro-F1, mean ± SD | Pooled macro-F1 | Pooled accuracy | Pooled balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 0.3485 ± 0.0183 | 0.3491 | 0.5774 | 0.3293 |
| Lasso | 0.3487 ± 0.0184 | 0.3492 | 0.5781 | 0.3295 |
| Elastic Net | 0.3489 ± 0.0185 | 0.3495 | 0.5782 | 0.3297 |

## Independent reproduction

Fiza's GaussianNB result was rerun using the experiment branch revision
`8d77a242d129137afe32f6dab151b848ed03b030`, the same verified NHAMCS file,
and the code block in that revision's `docs/gaussian_nb_experiment.md`.
The command loads the modelling inputs and calls
`run_evaluation(inputs, model_ids=["gaussian_nb"])` with the documented
five hospital-grouped folds, seed 42, and both feature conditions. The
rerun matched the scikit-learn pin. It used Linux and Python 3.12.3,
while the reference lists Python 3.14.2.

| Feature condition | Fold-mean metric | Published | Observed | Observed minus published |
| --- | --- | ---: | ---: | ---: |
| full | Macro-F1 | 0.1585 | 0.1585185693 | +0.0000185693 |
| full | Accuracy | 0.1626 | 0.1626365900 | +0.0000365900 |
| full | Balanced accuracy | 0.3209 | 0.3208551773 | -0.0000448227 |
| full | Macro ROC-AUC | 0.5699 | 0.5698647376 | -0.0000352624 |
| no_rfv | Macro-F1 | 0.0987 | 0.0986555690 | -0.0000444310 |
| no_rfv | Accuracy | 0.1151 | 0.1150607468 | -0.0000392532 |
| no_rfv | Balanced accuracy | 0.2864 | 0.2864260998 | +0.0000260998 |
| no_rfv | Macro ROC-AUC | 0.5960 | 0.5960432006 | +0.0000432006 |

All eight fold means match at the published four-decimal precision.
