# Part 3 nested tuning protocol

Issue #11 adds a reusable nested hyperparameter-tuning harness without changing
Part 2. `src.evaluation.runner.run_evaluation`, `EvaluationResult`, the six
Part 2 model/condition combinations, the Part 2 figure, and the notebook are
unchanged.

## Purpose

Part 2 reports cross-validated performance for fixed estimators. It does not
select hyperparameters, and a single cross-validation loop cannot be used for
both selection and reporting without optimistic bias. The nested harness
separates those jobs:

1. For each outer fold, run a hospital-grouped inner search on the
   outer-training rows only.
2. Select the best candidate by mean inner macro-F1.
3. Refit that candidate on the whole outer-training partition.
4. Score the held-out outer test fold exactly once.

The outer predictions are exact-once out-of-fold predictions suitable for the
Part 3 comparison.

## Splitting

| Stage | Splitter | Folds | Seed | Group |
| --- | --- | ---: | ---: | --- |
| Outer | `StratifiedGroupKFold` | 5 | 42 | `HOSPCODE` |
| Inner | `StratifiedGroupKFold` | 3 | 42 | `HOSPCODE` |

Both stages reuse the existing `src.evaluation.splits.make_grouped_folds`.
The inner splitter is fitted only on the corresponding outer-training
partition; its positions are translated back to absolute dataset positions and
stored in `NestedInnerFold`. No frozen hospital test set is used in this issue:
every hospital can appear in an outer test fold, but never in the training or
validation partitions of that same fold.

The harness checks that:

- outer train/test hospitals and positions are disjoint;
- inner train/validation hospitals and positions are disjoint;
- inner validation positions and hospitals never come from the outer test fold;
- outer test positions cover every row exactly once.

## Selection rule

The default scorer is a new additive helper,
`src.evaluation.metrics.make_macro_f1_scorer()`. It calls `f1_score` with:

- `average="macro"`;
- `labels=(1, 2, 3, 4, 5)`;
- `zero_division=0`.

This matches the Part 2 metric convention without altering the Part 2 metric
functions. A caller may pass another scorer through the `scoring` argument.

Candidates are generated with `sklearn.model_selection.ParameterGrid`. For each
candidate, the mean inner score across the three inner folds is computed. The
candidate with the highest mean wins; ties are broken deterministically by
selecting the first candidate in the actual `ParameterGrid` iteration order.
`np.argmax` returns the first maximum, so this tie-break follows the order in
which `ParameterGrid` yields candidates, not necessarily the order in which
dictionary keys were declared.

The outer fit uses a fresh clone of the same preprocessing pipeline and the
selected parameters. Its predictions contribute one row per held-out position
to `oof_predictions`. With two candidates the expected fit count is:

```
5 outer folds * (3 inner folds * 2 candidates + 1 final refit) = 35 fits
```

## API

```python
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import NestedSpec, prepare_modelling_inputs, run_nested_evaluation

inputs = prepare_modelling_inputs(population)
spec = NestedSpec(
    name="decision_tree_tuned",
    estimator_factory=lambda: DecisionTreeClassifier(random_state=42),
    param_grid={"estimator__max_depth": [3, 5]},
    condition="full",
)
result = run_nested_evaluation(inputs, spec)
```

Grid keys use the pipeline's parameter paths, for example
`estimator__max_depth` or `estimator__class_weight`. This is intentional:
estimator-native `class_weight` values are ordinary grid parameters and do not
require a special harness path.

`run_nested_evaluation` returns a dedicated `NestedEvaluationResult`, not the
Part 2 `EvaluationResult`. It contains:

- `folds`: the five outer `GroupedFold` objects;
- `fold_results`: one `NestedFoldResult` per outer fold, including the three
  `NestedInnerFold` objects, all inner scores, selected parameters, outer
  metrics, and outer predictions;
- `fold_metrics` and `summary`: Part 2-compatible metric columns and sampling
  standard deviations (`ddof=1`);
- `oof_predictions`, `pooled_oof_metrics`, `per_class_metrics`, and
  `confusion_matrix`: exact-once outer-fold diagnostics;
- `best_params`, `inner_search`, and `diagnostics`: selection audit trail;
- `metadata`: seeds, versions, fit counts, scorer, error policy, and weight
  policy;
- `fitted_pipelines`: populated only when `retain_fitted_pipelines=True`.

## Sample weights

Sample weighting is optional. The harness supports both a fixed policy and
selection among multiple candidate policies.

Part 3 provides inverse-frequency weighting:

```
w_c = n / (K * n_c)
```

where `n` is the number of rows in the current fit partition, `K = 5` is
the number of labels, and `n_c` is the count of class `c` in that
partition. The policy is implemented by `inverse_frequency_weights` and
exposed as `INVERSE_FREQUENCY_WEIGHT_POLICY`.

Weights are recomputed for every fit from only the rows used for that fit:
an inner-training partition during inner search, or the full outer-training
partition during the final refit. Outer test labels and features are never
used by the policy. `build_weighted_fit_params` routes weights to the
pipeline's final estimator as `estimator__sample_weight`; no preprocessor
step receives them. A `WeightPolicy` may override `fit_param` for a
different estimator-level argument, but pipeline paths are rejected.

A fixed policy can be supplied through `weight_policy`:

```python
from src.evaluation import INVERSE_FREQUENCY_WEIGHT_POLICY, NestedSpec

weighted_spec = NestedSpec(
    name="decision_tree_weighted",
    estimator_factory=lambda: DecisionTreeClassifier(random_state=42),
    param_grid={"estimator__max_depth": [3, 5]},
    weight_policy=INVERSE_FREQUENCY_WEIGHT_POLICY,
)
```

Alternatively, callers can compare candidate policies through
`weight_policies`. The unweighted option is represented by `None`:

```python
spec = NestedSpec(
    name="xgboost",
    estimator_factory=make_xgboost_estimator,
    param_grid={"estimator__max_depth": [3, 5]},
    condition="full",
    weight_policies=(None, INVERSE_FREQUENCY_WEIGHT_POLICY),
)
```

Each parameter combination is evaluated with each candidate weighting policy
using the same inner folds. The policy and hyperparameters are selected
together by mean inner macro-F1. For every inner fit, weights are calculated
only from that inner training partition. The selected policy is then
recomputed from the complete outer-training partition for the outer refit.
The outer test fold is not involved in either selection or weight calculation.

When `weight_policy` is `None` and no candidate list is supplied, the
fit is unweighted. The selected policy is recorded per outer fold in
`best_params` and the nested results. Metadata records the candidate
policies and the selected policy or policies.

A specification may instead provide `weight_policy_candidates`, a sequence of
`WeightPolicy` values and/or `None`. The harness evaluates each weighting
strategy as an inner-search candidate alongside each parameter-grid candidate,
and refits the inner-selected strategy on the complete outer-training
partition. This allows models without native `class_weight` support to compare
unweighted and inverse-frequency-weighted fits without consulting outer test
results. The selected policy is recorded for every outer fold; candidate policy
names are included in `inner_search` and run metadata.
Candidate order is parameter-grid order followed by the declared policy order;
the existing first-maximum tie break therefore prefers the first policy when
scores tie.

## Determinism and failure handling

- The harness is sequential: `search_n_jobs` is accepted for API clarity but
  must be `1`. This keeps inner fold order and tie-breaking reproducible.
- Randomness defaults to `config.RANDOM_STATE` (`42`) for both outer and inner
  splitting.
- `error_score="raise"` is the default. Any failed inner fit or score aborts
  the run with the outer fold, inner fold, and candidate identified. A finite
  numeric `error_score` is also accepted for deliberate robustness testing.
- `n_outer_splits` and `n_inner_splits` must each be at least 2.
- Static specification/configuration checks, such as unknown grid parameters,
  empty grids, invalid feature conditions, invalid split counts, and disallowed
  `search_n_jobs` values, run before fitting; supplied outer folds also receive
  structural validation via `validate_folds()`. Partition-specific checks,
  including malformed weights generated for a given fitting partition and
  per-fold position/hospital checks encountered during execution, occur when
  that partition or fold is processed. The harness does not transactionally
  pre-validate every possible runtime input before the first fit.

## Scope

This harness does not add a PyTorch/MLP adapter, does not tune preprocessing,
and does not change the Part 2 results pipeline. MLP support belongs to the
later model-specific issue. Part 3 can consume this harness for the decision
tree and any future estimator that satisfies the scikit-learn estimator
protocol.

## Verification

Run all non-integration tests:

```
python -m pytest -q -p no:cacheprovider
```

Run dataset-backed checks when the pinned NHAMCS file is available:

```
python -m pytest -m integration -q
```

The integration suite includes a nested smoke run with two candidates (35
nested fits), a Part 2 regression run (30 fits), and the byte-level hash check
for `reports/figures/fig05_part2_macro_f1_comparison.png`. The expected figure
SHA-256 is
`55AAC5DF03B159F4EE9465691E139171C68C70D954A3CAFF944627B013D8833F`.
