# Gini Decision Tree: results and interpretation (Issue #3)

Author: Romesa Irfan. AI-assisted: drafted with Claude from the experiment
outputs; numbers were produced by the scripts listed at the end and reviewed
by the author.

## 1. Scope

One model, one experiment: `DecisionTreeClassifier(criterion="gini",
random_state=42)` evaluated through the shared evaluator on the shared
hospital-grouped folds, in two feature conditions (`full` and `no_rfv`).
No tuning experiment was run.

## 2. Configuration

| Item | Value |
|---|---|
| Model id | `decision_tree_gini` (factory in `src/evaluation/runner.py`) |
| Estimator | `DecisionTreeClassifier(criterion="gini", random_state=42)`; all other parameters are sklearn defaults, so the tree grows without a depth limit |
| scikit-learn | 1.9.1 — exact pin in `requirements.txt`; fold assignments are version-dependent (see section 10) |
| Conditions | `full` and `no_rfv` (`FEATURE_CONDITIONS` in `src/evaluation/data.py`) |
| Folds | 5 grouped folds on `HOSPCODE` from `make_grouped_folds`, default arguments, generated once and reused |
| Preprocessing | Shared `ColumnTransformer` pipeline, refitted on the training rows of every fold |
| Population | 10,207 visits, 147 hospitals, classes 1 to 5 |

## 3. Fold diagnostics

| Fold | n_train | n_test | train hospitals | test hospitals | hospital overlap | test class 1 | test class 5 |
|---|---|---|---|---|---|---|---|
| 0 | 8167 | 2040 | 116 | 31 | 0 | 25 | 61 |
| 1 | 8168 | 2039 | 116 | 31 | 0 | 27 | 63 |
| 2 | 8176 | 2031 | 121 | 26 | 0 | 24 | 61 |
| 3 | 8143 | 2064 | 119 | 28 | 0 | 34 | 61 |
| 4 | 8174 | 2033 | 116 | 31 | 0 | 29 | 61 |

No hospital appears in both train and test of a fold, and all five classes
are present in every test fold. Class 1 has only 24 to 34 test rows per fold,
so its per-fold scores are noisy.

## 4. Results

Mean and sample standard deviation over the 5 folds:

| Metric | full | no_rfv |
|---|---|---|
| Macro-F1 (primary) | 0.2799 +/- 0.0198 | 0.2456 +/- 0.0151 |
| Micro-F1 (= accuracy) | 0.4627 +/- 0.0075 | 0.4013 +/- 0.0095 |
| Weighted-F1 | 0.4593 +/- 0.0070 | 0.4033 +/- 0.0100 |
| Balanced accuracy | 0.2774 +/- 0.0195 | 0.2452 +/- 0.0138 |
| ROC-AUC | 0.5509 +/- 0.0104 | 0.5269 +/- 0.0087 |

Micro-F1 equals accuracy in every fold (checked in `fold_metrics.py`); for
single-label multiclass data this is expected.

Pooled out-of-fold (all 10,207 rows): full macro-F1 0.2800, accuracy 0.4627,
balanced accuracy 0.2778; no_rfv macro-F1 0.2450, accuracy 0.4013, balanced
accuracy 0.2447.

Per-fold values for every metric and per-class F1 are in
`docs/decision_tree_fold_metrics.csv`.

| Condition | Fold | Macro-F1 | Micro-F1 | ROC-AUC |
|---|---|---|---|---|
| full | 0 | 0.3035 | 0.4686 | 0.5646 |
| full | 1 | 0.2676 | 0.4625 | 0.5462 |
| full | 2 | 0.2529 | 0.4510 | 0.5365 |
| full | 3 | 0.2867 | 0.4617 | 0.5531 |
| full | 4 | 0.2889 | 0.4697 | 0.5541 |
| no_rfv | 0 | 0.2315 | 0.3873 | 0.5184 |
| no_rfv | 1 | 0.2421 | 0.4061 | 0.5283 |
| no_rfv | 2 | 0.2714 | 0.4131 | 0.5412 |
| no_rfv | 3 | 0.2417 | 0.3997 | 0.5226 |
| no_rfv | 4 | 0.2415 | 0.4004 | 0.5242 |

Reference points (arithmetic, not from a run in this experiment): the
majority-class baseline has accuracy 0.5232 (class 3 share). Predicting class
3 for every visit would give macro-F1 of about 0.137. The tree's accuracy is
below the majority baseline, while its macro-F1 is above that level.

ROC-AUC is computed from probabilities aligned to the fixed label order by the
shared evaluator (`align_via_pipeline` -> `align_proba_columns`). This
experiment does not modify the evaluator. ROC-AUC was available for all 10
fold-condition runs.

## 5. Confusion matrices and rare classes

Pooled out-of-fold confusion matrices (rows = true class, columns = predicted).

full:

| true \ pred | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 11 | 38 | 68 | 21 | 1 |
| 2 | 39 | 442 | 826 | 264 | 24 |
| 3 | 51 | 755 | 3132 | 1278 | 124 |
| 4 | 17 | 224 | 1366 | 1123 | 96 |
| 5 | 0 | 34 | 135 | 123 | 15 |

no_rfv:

| true \ pred | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 15 | 41 | 59 | 21 | 3 |
| 2 | 35 | 344 | 793 | 382 | 41 |
| 3 | 67 | 852 | 2761 | 1471 | 189 |
| 4 | 12 | 352 | 1409 | 963 | 90 |
| 5 | 0 | 50 | 154 | 90 | 13 |

Per-class recall (diagonal over row total):

| Class | n | full | no_rfv |
|---|---|---|---|
| 1 Immediate | 139 | 11 (7.9%) | 15 (10.8%) |
| 2 Emergent | 1,595 | 442 (27.7%) | 344 (21.6%) |
| 3 Urgent | 5,340 | 3132 (58.7%) | 2761 (51.7%) |
| 4 Semi-urgent | 2,826 | 1123 (39.7%) | 963 (34.1%) |
| 5 Non-urgent | 307 | 15 (4.9%) | 13 (4.2%) |

Rare classes:

- Class 1: the tree finds only 11 (full) and 15 (no_rfv) of 139 visits.
  Precision is 9.3% (11 of 118 predicted) in full and 11.6% (15 of 129) in
  no_rfv. Per-fold F1 for class 1 ranges from 0.000 to 0.200 in full and
  0.068 to 0.186 in no_rfv; it is exactly 0.000 in folds 1 and 2 of full.
  With only 24 to 34 class-1 rows per test fold, these differences are
  within what noise can produce, so we do not rank the two conditions on
  class 1.
- Class 5: 15 (full) and 13 (no_rfv) of 307 are found. Per-fold F1 stays
  between 0.030 and 0.108 in every fold of both conditions.
- Most errors are to adjacent or near classes. In full, 76% of class-1 visits
  (106 of 139) are predicted as 2 or 3, and 84% of class-5 visits (258 of
  307) are predicted as 3 or 4.
- Mean absolute error in acuity levels, computed from the pooled confusion
  matrices above: 0.641 (full) and 0.742 (no_rfv). This is a descriptive
  ordinal view only; the primary metric remains macro-F1.

## 6. Full versus no_rfv

- Accuracy: full is higher in all 5 folds (0.4627 vs 0.4013 on average).
- Macro-F1: full is higher in 4 of 5 folds; in fold 2 no_rfv is higher
  (0.2714 vs 0.2529). The mean difference is about 0.034, which is less than
  two fold standard deviations (0.020 and 0.015), so this is a modest
  difference rather than a clear separation.
- Balanced accuracy and ROC-AUC: full is higher in 4 of 5 folds; no_rfv is
  higher in fold 2 in both.
- Class 2, 3 and 4 recall are higher in full (section 5); class 1 recall is
  higher in no_rfv, but see the noise caveat above.
- Both conditions have ROC-AUC close to 0.5 (0.551 and 0.527).

RFV is retained under the disclosed assumption in the README: the released
4-digit codes were assigned retrospectively by RTI. The `full` condition
therefore carries that assumption, and `no_rfv` is the check that does not.

## 7. Interpretability (all five folds)

The fitted tree for each fold and condition is inspected from the pipeline
retained by the shared `run_evaluation` call; no folds are regenerated and no
trees are refitted for these diagnostics. Script:
`scripts/tree_splits_all_folds.py` calls `run_evaluation` with
`retain_fitted_pipelines=True`, then reads the fitted pipelines keyed by model,
condition and fold. The held-out scores come from the evaluator's fold metrics.
Outputs:
`docs/decision_tree_all_folds_tree_summary.csv`,
`docs/decision_tree_all_folds_splits.txt` (text export to depth 3 for every
fold and condition).

Tree structure per fold:

| Condition | Fold | n_train | Depth | Leaves | Train rows per leaf | Held-out macro-F1 |
|---|---|---|---|---|---|---|
| full | 0 | 8167 | 67 | 2460 | 3.32 | 0.3035 |
| full | 1 | 8168 | 87 | 2454 | 3.33 | 0.2676 |
| full | 2 | 8176 | 64 | 2493 | 3.28 | 0.2529 |
| full | 3 | 8143 | 74 | 2462 | 3.31 | 0.2867 |
| full | 4 | 8174 | 79 | 2495 | 3.28 | 0.2889 |
| no_rfv | 0 | 8167 | 42 | 2816 | 2.90 | 0.2315 |
| no_rfv | 1 | 8168 | 36 | 2788 | 2.93 | 0.2421 |
| no_rfv | 2 | 8176 | 40 | 2772 | 2.95 | 0.2714 |
| no_rfv | 3 | 8143 | 39 | 2813 | 2.89 | 0.2417 |
| no_rfv | 4 | 8174 | 45 | 2781 | 2.94 | 0.2415 |

- Depth ranges over 64 to 87 (full) and 36 to 45 (no_rfv). Leaf counts range
  over 2454 to 2495 (full) and 2772 to 2816 (no_rfv). The no_rfv trees are
  shallower but have more leaves.
- Across these five folds there is no visible relationship between depth or
  leaf count and held-out macro-F1 (for example fold 1 has the deepest full
  tree and the second-lowest macro-F1; fold 2 has the shallowest full tree
  and the lowest). This was not tested statistically.

Root and first-level splits (thresholds are in standardized units). Names in
this table are abbreviated for width; the legend below maps each displayed
name to the exact transformed feature name used by the pipeline, with its raw
NHAMCS source column. `amb_1` and `amb_2` are two one-hot columns of the same
arrival-by-ambulance variable; we did not check whether they are exact
complements:

| Condition | Fold | Root | Left child | Right child |
|---|---|---|---|---|
| full | 0 | rfv1_broad_category_1545 | amb_1 | temp_f <= 1.99 |
| full | 1 | rfv1_broad_category_1545 | amb_2 | resp_rate <= 0.30 |
| full | 2 | rfv1_broad_category_1545 | amb_2 | resp_rate <= 0.33 |
| full | 3 | rfv1_broad_category_1545 | amb_2 | resp_rate <= 0.33 |
| full | 4 | rfv1_broad_category_1545 | amb_1 | resp_rate <= 0.49 |
| no_rfv | 0 | amb_1 | missingindicator_bp_systolic <= 1.44 | resp_rate <= 0.12 |
| no_rfv | 1 | amb_2 | bp_systolic <= -2.69 | bp_systolic <= -0.07 |
| no_rfv | 2 | amb_1 | missingindicator_bp_systolic <= 1.64 | resp_rate <= 0.16 |
| no_rfv | 3 | amb_2 | resp_rate <= 0.16 | missingindicator_bp_systolic <= 1.56 |
| no_rfv | 4 | amb_1 | missingindicator_bp_systolic <= 1.52 | resp_rate <= 0.16 |

Feature-name legend — displayed abbreviation, exact transformed feature name,
origin:

| Displayed | Exact transformed feature name | Origin |
|---|---|---|
| `amb_1` | `categorical_low__arrival_by_ambulance_1` | one-hot level 1 of engineered column `arrival_by_ambulance` (raw `ARREMS`) |
| `amb_2` | `categorical_low__arrival_by_ambulance_2` | one-hot level 2 of the same engineered column (raw `ARREMS`) |
| `rfv1_broad_category_1545` | `categorical_high__rfv1_broad_category_1545` | one-hot level 1545 of engineered column `rfv1_broad_category` (raw `RFV13D`, first of the five 4-digit reason-for-visit slots) |
| `temp_f` | `numeric__temp_f` | engineered `temp_f` (raw `TEMPF`, divided by 10) |
| `resp_rate` | `numeric__resp_rate` | engineered `resp_rate` (raw `RESPR`) |
| `bp_systolic` | `numeric__bp_systolic` | engineered `bp_systolic` (raw `BPSYS`) |
| `missingindicator_bp_systolic` | `numeric__missingindicator_bp_systolic` | missingness indicator created inside each training fold by `SimpleImputer(add_indicator=True)` for `bp_systolic`; it has no raw source column of its own |

The `numeric__`, `categorical_low__` and `categorical_high__` prefixes are the
block names in `build_column_transformer` (`verbose_feature_names_out=True`,
`src/preprocessing/features.py`). Transformed names are not raw NHAMCS
columns; the raw sources are listed only to identify where each measurement or
code comes from. The artifacts `docs/decision_tree_all_folds_tree_summary.csv`
and `docs/decision_tree_all_folds_splits.txt` store the full transformed
names, not the abbreviations used in the table above.

- full: the root is the same column in all five folds. The left child is the
  ambulance variable in all five. The right child is temp_f in fold 0 only
  and resp_rate in folds 1 to 4.
- no_rfv: the root is the ambulance variable in all five folds (amb_1 in
  folds 0, 2, 4; amb_2 in folds 1, 3). Folds 0, 2 and 4 use the same two
  child features (thresholds differ slightly); fold 3 uses the same two
  features on opposite sides; fold 1 differs (bp_systolic on both sides). A
  missingness indicator appears at the first level in 4 of 5 no_rfv folds; we
  report it as an observation and give no explanation.
- Agreement deeper than the first level was not examined; the text export
  to depth 3 for every fold is in `docs/decision_tree_all_folds_splits.txt`.

Gini importances:

- In every fold of both conditions the top five columns are the same five
  numeric ones (arrival_minutes, bp_systolic, bp_diastolic, pulse_bpm,
  temp_f); only their order changes.
- Columns in the top ten of all five folds: full has 8 (the seven numeric
  columns and rfv1_broad_category_1545); no_rfv has 7 (all seven numeric
  columns).
- Fold 0 values: full arrival_minutes 0.086, bp_systolic 0.082, pulse_bpm
  0.076, bp_diastolic 0.068, temp_f 0.064; no_rfv arrival_minutes 0.148,
  bp_diastolic 0.124, pulse_bpm 0.113.

Reading these results:

- Numeric thresholds are in standardized units (the pipeline scales numeric
  columns on the training fold), not in original units.
- RFV is one-hot encoded, so its importance is spread over many columns and
  each one looks small.
- Gini importance favours columns with many possible split points, such as
  continuous measurements. It should not be read as evidence that vitals
  carry more information than RFV. The audit finds the vitals only weakly
  associated with the target.
- Trees with thousands of leaves cannot be read as rule sets; only the top
  levels are interpretable.

## 8. Overfitting

What is observed:

- The trees have no depth or leaf-size limit. Across the five folds they
  reach depth 64 to 87 and 2454 to 2495 leaves (full), and depth 36 to 45
  and 2772 to 2816 leaves (no_rfv), on about 8,150 training rows each: about
  3.3 (full) and 2.9 (no_rfv) training rows per leaf.
- Held-out macro-F1 is 0.28 (full) and 0.25 (no_rfv), and held-out ROC-AUC is
  0.55 and 0.53.

What is not measured: the shared runner reports held-out scores only, and no
training-set score was computed. We therefore make no numeric claim about a
train versus held-out gap. The structure above is consistent with overfitting
but does not demonstrate it. A diagnostic train-score fit (same configuration,
no tuning) was not run because it needs team agreement; if approved, it
would close this gap.

## 9. Limitations

- RFV assumption: the codes were assigned retrospectively (2.4% recoding error
  reported in the README), so `full` results depend on it.
- Rare classes: 139 class-1 and 307 class-5 visits in total, 24 to 34 and 61
  to 63 per test fold; per-class and per-fold results for them are unstable.
- Weak signal: ROC-AUC is close to 0.5, and a single unpruned tree gives
  coarse probabilities, which also limits what ROC-AUC can show (a general
  property of unpruned trees, not tested separately here).
- Split-level comparison covers the root and first level (top three levels
  in the text export); agreement deeper in the trees was not examined.
- Five folds, 26 to 31 test hospitals per fold: fold-to-fold variation is
  visible in the standard deviations and the differences between conditions
  are small relative to it.
- No tuning, no class weighting, no other models were run here.

## 10. Reproduction

From the repository root, with `data/local/ed2022-stata.dta` in place:

    python -m pytest tests/test_integration_real_data.py -m integration -q
    python -m scripts.run_decision_tree      # summary, pooled, per-class, confusion
    python -m scripts.fold_metrics           # per-fold table and CSV
    python -m scripts.tree_splits_all_folds  # shared run, all folds' diagnostics

Seed: `RANDOM_STATE = 42` (`src/config.py`).

### Reproducibility environment

The experiment uses `scikit-learn 1.9.1` (with pandas 2.3.3, NumPy 2.5.4,
SciPy 1.18.1), pinned exactly in `requirements.txt` as `scikit-learn==1.9.1`.

`StratifiedGroupKFold` fold assignments can differ between scikit-learn
versions even when `shuffle=True` and `random_state=42` are unchanged, because
the splitter is library code whose algorithm may change between releases; the
seed fixes the randomness within one implementation, not across versions. The
fold-level results above — fold sizes, per-fold metrics, and the fitted trees'
splits — should therefore be read against the pinned version: re-running under
a different scikit-learn version may produce different folds and different
metrics from the same seed. The pin removes that source of variation for the
project; it does not by itself guarantee byte-identical results across
operating systems, Python versions, or numerical-library builds.

## 11. Open items

- Trained-tree splits were produced for every shared fold and compared
  (section 7).
- Whether a diagnostic train-score fit is allowed (section 8).
