# Model experiment guide

This guide is the single entry point for running, reporting, and reviewing the
Part 3 and Part 4 model experiments. It consolidates the shared dataset,
evaluation protocol, feature conditions, reporting schema, contributor
responsibilities, and workflow rules so every experiment is reproducible and
directly comparable. It does not duplicate the protocol specification: the
normative definition of nested tuning lives in
[`nested_tuning_protocol.md`](nested_tuning_protocol.md), and the supported
Python environment lives in [`environment.md`](environment.md).

## Dataset and target

- **Dataset:** 2022 NHAMCS Emergency Department public-use file,
  `ed2022-stata.dta` (expected size 37,720,094 bytes, expected shape
  `(16,025, 913)`). The repository never stores the file; see
  [`../data/README.md`](../data/README.md) for acquisition and verification,
  and [`environment.md`](environment.md) for path resolution
  (`NHAMCS_DATA_PATH` or `data/local/ed2022-stata.dta`).
- **Target:** `IMMEDR` (immediacy with which patient should be seen), modelled
  as a fixed five-class problem with labels `1..5`
  (`src.evaluation.data.LABELS`, from `src.data.target.URGENCY_CLASSES`).
  Metrics, probability columns, and result tables all use this fixed label
  order.
- **Population:** raw rows are filtered with
  `src.data.target.select_modelling_population`, then projected into aligned
  `X`, `y`, and `groups` by
  `src.evaluation.data.prepare_modelling_inputs`. The group column is
  `HOSPCODE` (`src.evaluation.data.GROUP_COLUMN`) and is used **only** for
  splitting; it must never appear as a predictive feature.
- **Standard load sequence** (identical in every script and experiment):

  ```python
  from src.data import dataset, target
  from src.evaluation import prepare_modelling_inputs

  full = dataset.load_raw_dataset(verify=True, expect_shape=True)
  population = target.select_modelling_population(full)
  inputs = prepare_modelling_inputs(population)
  ```

  `ModellingInputs` enforces index alignment, missing-group rejection, the
  approved feature register, and label containment before any split is made.

## Evaluation protocol

The normative specification is [`nested_tuning_protocol.md`](nested_tuning_protocol.md).
The rules every experiment must follow:

| Stage | Splitter | Folds | Seed | Group |
| --- | --- | ---: | ---: | --- |
| Outer | `StratifiedGroupKFold` | 5 | 42 | `HOSPCODE` |
| Inner | `StratifiedGroupKFold` | 3 | 42 | `HOSPCODE` |

- Seed `42` is `src.config.RANDOM_STATE`; when `random_state=None` the
  splitters default to it. Outer folds are generated once per run and shared
  across all model-condition combinations so results are directly comparable.
- **Selection:** for each outer fold, every grid candidate is scored on the
  three inner folds with the default scorer
  `make_macro_f1_scorer()` (`f1_score`, `average="macro"`,
  `labels=(1, 2, 3, 4, 5)`, `zero_division=0`). The highest mean inner score
  wins; ties are broken deterministically by the first candidate in
  `ParameterGrid` iteration order (`np.argmax` returns the first maximum).
- **Refit and score:** the winning candidate is refit on the whole
  outer-training partition and scored exactly once on the held-out outer test
  fold. Outer predictions contribute exactly one out-of-fold (OOF) row per
  dataset row.
- **No leakage:** preprocessing is fold-local (fit on the training partition
  only), inner validation positions and hospitals are always a subset of the
  outer-training partition, and outer test data is never used for selection,
  early stopping, or preprocessing.
- **Determinism:** `search_n_jobs` must be `1` (the harness is sequential by
  protocol); `error_score="raise"` is the default and any failed inner fit
  aborts the run with fold and candidate identified.
- **Spread:** reported `*_std` values are sample standard deviations with
  `ddof=1` over the five outer folds (`mean_std_summary`).
- Part 2 fixed-estimator baselines use `run_evaluation` on the same grouped
  folds; nested tuning (`run_nested_evaluation`) is the required path for any
  hyperparameter or weighting-strategy selection.

## Feature conditions

Two approved feature conditions exist
(`src.evaluation.data.FEATURE_CONDITIONS`); experiments select one by name
through `NestedSpec(condition=...)` or `run_evaluation(conditions=...)`:

| Condition | Columns | Meaning |
| --- | ---: | --- |
| `full` | 20 | The complete approved feature register (`src.preprocessing.features.FEATURES`). |
| `no_rfv` | 15 | `full` minus the canonical RFV complaint columns (`RFV_COLUMNS`). |

- `src.evaluation.data.feature_columns_for(condition)` returns the column
  tuple; `select_features(inputs, condition)` projects the matrix.
- `src.preprocessing.features.leakage_checks` runs at pipeline-construction
  time for every condition: the target and hospital/survey identifiers must
  never be present.
- The approved register is fixed. Experiments must not invent feature
  sets outside these two conditions; new conditions require an issue-level
  decision first.
- RFV exclusion is an **ablation**, not a preprocessing change: the
  preprocessing pipeline itself is identical for both conditions.

## Reporting requirements

Every experiment PR must commit the standard result schema, produced by the
shared runners - not hand-assembled tables.

**For `run_nested_evaluation` (`NestedEvaluationResult`):**

- `fold_metrics` - one row per outer fold with `model`, `condition`, `fold`,
  `macro_f1`, `accuracy`, `balanced_accuracy`, `roc_auc` (plus
  `roc_auc_available` / `roc_auc_reason` when a class is absent), `n_test`,
  and per-label `precision_k` / `recall_k` / `f1_k` for `k = 1..5`.
- `summary` - one row per model-condition with `n_folds` and
  `<metric>_mean`, `<metric>_std`, `<metric>_n_valid` for each metric in
  `METRIC_COLUMNS` (`macro_f1`, `accuracy`, `balanced_accuracy`, `roc_auc`).
- `oof_predictions` - exact-once outer-fold predictions (one row per dataset
  row), plus `pooled_oof_metrics` and `per_class_metrics` computed from them.
- `confusion_matrix` - rows = true labels, columns = predicted, labels 1..5.
- `best_params` and `inner_search` - the per-outer-fold selection audit trail
  (candidate params, inner scores, selected weighting strategy).
- `diagnostics` - outer/inner fold sizes, hospital counts, and hospital
  overlap checks.
- `metadata` - seeds, library versions, fit counts, scorer, error policy, and
  weight policy.
- `fitted_pipelines` - only when `retain_fitted_pipelines=True`; omit
  otherwise to keep artifacts small.

**Conventions:**

- Report mean +/- sample SD (`ddof=1`) across the five outer folds; never
  report a single-fold number as the headline result.
- Record best parameters and the weighting strategy per outer fold (they may
  legitimately differ across folds).
- Weighting comparisons (`class_weight='balanced'` vs inverse-frequency
  `sample_weight` vs unweighted) are decided **inside the inner CV only** and
  must be pre-declared in the PR description.
- No post-hoc condition switching: `no_rfv` results are produced only when
  declared in advance (e.g. as a family winner rule), not chosen after seeing
  results.
- Regenerate committed CSVs from code (e.g. `scripts/fold_metrics.py`
  patterns); never edit result files by hand.

## Contributor responsibilities

Assignment matrix as currently recorded on GitHub (verify before starting):

| Issue | Model / deliverable | Owner | Notes |
| --- | --- | --- | --- |
| #12 | Random Forest + AdaBoost experiments | Romaisa (`romaisairfan`) | Full condition for both models; `no_rfv` only under a pre-declared winner rule; balance strategies compared within inner CV. |
| #13 | XGBoost experiment | Fiza (`fizanoor0208`) | Install via `python -m pip install -r requirements-xgboost.txt`; never add `xgboost` to core `requirements.txt`. |
| #14 | Regularized Logistic Regression (L1 / L2 / Elastic Net) | Wahad (`mwa902`) | `saga` solver; document Ridge/Lasso/Elastic-Net mapping onto multiclass logistic regression; validate solver-penalty compatibility. |
| #16 | PyTorch MLP (Part 4) | **Unassigned - awaiting confirmation** | Confirmation comment posted on the issue; do not start until ownership is confirmed. Same outer folds; grouped early-stopping split drawn only from outer-training hospitals. |
| #17 | PCA + K-means analysis | Wahad (`mwa902`) | Numeric-only PCA primary; pre-declared k range and selection rule; never fit on `IMMEDR`; target-by-cluster only as labelled post-hoc description. |
| #18 | Interpretability + responsible-use limitations | Romaisa (`romaisairfan`) | Blocked by #12, #13, and #16 results; permutation importance is the floor; no demographic fairness or clinical-validity claims. |

Shared duties for every experiment owner:

1. Branch from the latest `origin/dev` and open the PR against `dev`.
2. Reuse the shared pipeline and runners; do not modify evaluation,
   preprocessing, splitting, metric, or model core code - request changes
   through an issue if the shared layer appears wrong.
3. Reproduce at least one other member's result before opening the PR
   (acceptance criterion on every experiment issue).
4. Run the verification commands below and include the output in the PR.
5. Add tests and documentation for new code; keep existing tests green.
6. Keep issue metadata (assignment, milestone, labels) accurate; do not
   reassign or close other people's issues.

## Running a nested experiment (verified example)

The following is a complete, code-verified example using the public API
exports of `src.evaluation`:

```python
from sklearn.tree import DecisionTreeClassifier

from src.data import dataset, target
from src.evaluation import NestedSpec, prepare_modelling_inputs, run_nested_evaluation

full = dataset.load_raw_dataset(verify=True, expect_shape=True)
population = target.select_modelling_population(full)
inputs = prepare_modelling_inputs(population)

spec = NestedSpec(
    name="decision_tree_tuned",
    estimator_factory=lambda: DecisionTreeClassifier(random_state=42),
    param_grid={"estimator__max_depth": [3, 5]},
    condition="full",
)
result = run_nested_evaluation(inputs, spec)

result.summary                      # mean/std/n_valid per metric
result.fold_metrics                 # one row per outer fold
result.best_params                  # selected params per outer fold
result.summary_for("decision_tree_tuned", "full")
```

Verified against `src/evaluation/nested.py` and `src/evaluation/__init__.py`:

- `NestedSpec(name, estimator_factory, param_grid, condition="full",
  weight_policy=None)`; `estimator_factory()` must return a fresh sklearn
  `BaseEstimator` on every call.
- `param_grid` must be a non-empty mapping whose keys are valid
  `pipeline.get_params(deep=True)` paths. The shared pipeline is
  `("preprocess", ...) -> ("estimator", ...)`, so estimator parameters are
  written **pipeline-prefixed**, e.g. `estimator__max_depth`,
  `estimator__class_weight`. Unknown keys fail before any fit with an error
  naming the correct convention.
- `run_nested_evaluation(inputs, spec, *, outer_folds=None,
  n_outer_splits=5, n_inner_splits=3, random_state=None, scoring=None,
  search_n_jobs=1, retain_fitted_pipelines=False, error_score="raise",
  verbose=0)` returns a `NestedEvaluationResult` (not the Part 2
  `EvaluationResult`).
- Optional sample weighting is opt-in via
  `weight_policy=INVERSE_FREQUENCY_WEIGHT_POLICY`; weights are recomputed per
  fit partition and routed to the final estimator only
  (`estimator__sample_weight`). See the "Sample weights" section of
  [`nested_tuning_protocol.md`](nested_tuning_protocol.md).

**PyTorch warning:** `NestedSpec` and `run_nested_evaluation` are
scikit-learn-only. `estimator_factory` must return a sklearn estimator and the
harness fits a sklearn `Pipeline`; there is **no PyTorch/MLP adapter** in the
nested harness (explicit scope limit of Issue #11). The Part 4 MLP (#16) must
be implemented as a separate, coordinated adapter that reuses the same outer
folds (`make_grouped_folds` / `validate_folds`) and the fold-local
preprocessing outputs, with grouped early stopping drawn only from
outer-training hospitals. Coordinate the adapter design with Haris before
implementation; do not fork or modify `src/evaluation/nested.py` to force a
PyTorch loop into the sklearn path.

## Environment and optional dependencies

- Core environment (Python, scikit-learn pin, required packages):
  [`environment.md`](environment.md) and the root `requirements.txt`.
- XGBoost (Issue #13): `python -m pip install -r requirements-xgboost.txt`.
  The optional manifest is deliberate; keep `xgboost` out of the core file.
- PyTorch CPU (Issue #16): `python -m pip install -r requirements-torch-cpu.txt`.
  The platform scope of that manifest is documented in
  [`environment.md`](environment.md).
- Record the environment used in the PR description (Python version, package
  pins from the manifests) so results remain reproducible.

## Git workflow

1. `git fetch origin` and confirm the latest `origin/dev`; never branch from
   a stale local copy and never commit directly to `dev`.
2. Create a topic branch, e.g. `feat/<owner>/issue-NN-<slug>` for experiments
   or `docs/<owner>/<slug>` for documentation.
3. Make the minimal change for one issue; unrelated fixes go to separate
   PRs.
4. Commit with a concise imperative message referencing the issue number.
5. Push the branch and open a PR **against `dev`** with: issue link, what
   changed, results summary (if any), and verification output.
6. Do not merge your own PR; merge happens after review. Do not amend or
   force-push shared history after review has started.
7. Result CSVs, figures, and notebooks are regenerated from code in the same
   PR that changes the producing code.

## Scientific limitations

State these honestly in reports and the final notebook; they bound what any
experiment may claim:

- **Single seed.** All splits use seed 42; reported fold variance is
  variation across hospital-grouped folds, not across random seeds. Do not
  present it as run-to-run instability, and document the single-seed
  limitation wherever results are summarised.
- **One dataset, one year.** All results are internal cross-validated
  performance on the 2022 NHAMCS ED file. There is no external validation
  site or year; do not generalise beyond this dataset.
- **Survey design not modelled.** NHAMCS is a complex survey; sample weights
  and design variables are not part of the predictive protocol, so estimates
  are not population-representative.
- **Metric caveats.** Macro-F1 over fixed labels 1..5 with
  `zero_division=0` can be depressed by folds where a class is rare or
  absent; `roc_auc` is marked unavailable (`roc_auc_available=False`) with a
  reason when a required class is missing from a partition - never silently
  average it away.
- **No clinical claims.** Models are methodological exercises on unimputed
  `IMMEDR`. Outputs must not be described as clinically validated, safe for
  triage use, or as reflecting care standards.
- **No demographic fairness claims.** `AGE`, `SEX`, and the race/ethnicity
  columns are excluded from the feature set as target-impaired leakage
  proxies; without them no demographic fairness audit is possible. Subgroup
  error analyses may only use non-protected proxies and must be labelled
  descriptive (Issue #18).
- **Confidence is not calibration.** Predicted probabilities must not be
  called calibrated without an actual calibration analysis.
- **Unsupervised outputs are descriptive.** PCA and K-means (#17) never use
  the target for fitting; target-by-cluster tables are post-hoc description
  only, and cluster profiles are not clinical entities. Euclidean K-means on
  mixed feature types has known limitations - discuss them.
- **Single-model wins are provisional.** Model comparisons are only valid on
  identical outer folds with identical conditions; never compare numbers
  produced by different split sets or protocols.

## Verification

Before opening a PR, run from the repository root:

```bash
python -m pytest -q -p no:cacheprovider          # all non-integration tests
python -m pytest -m integration -q               # dataset-backed checks (requires the NHAMCS file)
```

The integration suite includes the nested smoke run (two candidates, 35
fits), the Part 2 regression run (30 fits), and the byte-level figure hash
check described in [`nested_tuning_protocol.md`](nested_tuning_protocol.md).
All non-integration tests must pass, and integration tests must pass when the
dataset is available; state clearly in the PR if they could not be run.

## Related documents

- [`nested_tuning_protocol.md`](nested_tuning_protocol.md) - normative
  nested CV specification (splits, selection rule, weights, determinism).
- [`environment.md`](environment.md) - supported Python environment and
  optional dependency manifests.
- [`feature_register.md`](feature_register.md) - approved feature list and
  leakage rationale.
- [`methodology.md`](methodology.md) - overall study design and decisions.
