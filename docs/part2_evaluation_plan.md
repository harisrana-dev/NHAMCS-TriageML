# Part 2 Evaluation Plan

**Status:** Design proposal — no Part 2 models have been trained and no model-performance results are claimed.

## 1. Objective

Compare Gaussian Naive Bayes and Decision Tree classifiers for the five-class NHAMCS emergency-department triage target, while estimating generalization to hospitals not represented in model training.

The evaluation framework is owned by the project lead and is shared by both model implementations. Models must use identical outer folds and metric definitions.

## 2. Frozen task definition

- **Dataset:** 2022 NHAMCS Emergency Department public-use file, verified by the Part 1 loader.
- **Population:** rows with genuine IMMEDR labels 1–5 only (10,207 rows in the verified Part 1 dataset).
- **Target:** IMMEDR, with class codes 1, 2, 3, 4, 5 preserved.
- **Features:** the approved 20 engineered columns from Part 1.
- **Grouping variable:** HOSPCODE, retained separately for splitting and audit only; it must never enter the predictive feature matrix.
- **Preprocessing:** reuse the Part 1 Pipeline/ColumnTransformer. Every learned preprocessing step must be fitted within each training fold.
- **Random seed:** use the project seed RANDOM_STATE = 42; record all additional seeds if repeated splits are used.

Do not alter target inclusion rules or add post-triage, target-derived, hospital-identifier, or target-imputed demographic features as part of model implementation.

## 3. Validation design

### Primary design

Use **5-fold StratifiedGroupKFold**, with HOSPCODE as groups, shuffling enabled, and a fixed seed. Repeat the five-fold process with a small, predeclared set of seeds (initial proposal: 42, 52, 62) to examine split sensitivity. The exact number of repeats is conditional on the fold audit and runtime.

This estimates performance on held-out hospitals, rather than merely on new visits from hospitals already represented in training.

### Mandatory split audit before fitting models

For every fold and repeat, record and check:

1. No HOSPCODE appears in both training and validation.
2. Training and validation row counts and distinct hospital counts.
3. Class counts in both partitions.
4. Whether each of the five classes occurs in the training partition and validation partition.
5. The proportion of the full dataset and of each class assigned to each validation fold.

StratifiedGroupKFold attempts to preserve class proportions but cannot guarantee perfect stratification when groups are indivisible. Do not assume class coverage: inspect the generated folds first. If a class is absent from a validation fold, document it and use explicit labels=[1, 2, 3, 4, 5] when computing class metrics. If a class is absent from a training fold, flag that fold because the model cannot learn that class. Reconsider the fold count only after reviewing this audit; do not change the split strategy just to improve scores.

### Leakage boundary

For each fold, pass the raw approved feature columns to a **fresh, unfitted pipeline**. Call fit only on the training rows, then predict the held-out rows. Never fit an imputer, scaler, encoder, feature selector, or category-frequency rule on all 10,207 rows before cross-validation.

HOSPCODE is passed to the splitter separately and excluded from X. Use the same saved split indices for every model and the baseline.

## 4. Baseline and metric correction

The value **0.5232 is the prevalence of the majority class (IMMEDR = 3)** in the modelling population. It is also the accuracy of a constant majority-class predictor evaluated against the full population. It is **not** the Macro-F1 score of that predictor.

The baseline for Part 2 must be a DummyClassifier(strategy="most_frequent") evaluated on the same grouped folds as the candidate models. Fit the dummy classifier on each training fold and score it on that fold's held-out hospitals. Report its observed Macro-F1 and secondary metrics from the evaluation run; do not infer or invent them from the class prevalence.

## 5. Metrics and reporting

### Primary metric

- **Macro-F1**, computed over the fixed label set [1, 2, 3, 4, 5] with zero_division=0 for consistent class handling.

### Secondary metrics

- Accuracy.
- Balanced accuracy.
- Per-class precision, recall, and F1 for all five classes.
- Confusion matrix with fixed class ordering [1, 2, 3, 4, 5].
- Class support counts, so low-support classes are visible.

The Immediate class (1) is rare, so never report accuracy alone or hide per-class results behind a single average. Keep any ordinal/under-triage analysis explicitly secondary and define its direction correctly: predicting a numerically higher class code than the true class means predicting a less urgent level.

### Aggregation

For each repeat, save exactly one out-of-fold prediction per row and compute a pooled out-of-fold report across all rows for that repeat. Also retain fold-level Macro-F1 and its descriptive mean and standard deviation. Report repeat-level pooled scores and fold-level variation transparently.

Do not treat overlapping cross-validation folds as independent samples or use fold standard deviation as a formal confidence interval. Show confusion matrices and per-class results from pooled out-of-fold predictions, with the repeat used clearly identified.

## 6. Model comparison rules

- Evaluate Gaussian Naive Bayes and Decision Tree using the same split indices and approved features.
- Each estimator must be inside the same fold-local preprocessing boundary.
- Record model parameters and library versions.
- Any tuning must occur using training data only; the held-out fold must not guide parameter selection.
- Report the dummy baseline beside both models.
- Do not claim that a model is superior unless the metric evidence supports the claim; explain class-level trade-offs, not only the highest score.
- Gaussian Naive Bayes requires dense input in scikit-learn. Check the ColumnTransformer output type and handle densification within the fold-local model pipeline if needed; never densify or transform the complete dataset using statistics learned globally.

## 7. Reproducibility artifacts

The evaluation implementation should save or expose:

- Split audit table.
- Fold-level metric table.
- Per-repeat pooled out-of-fold predictions.
- Per-class classification reports and confusion matrices.
- A model-comparison table including the dummy baseline.
- Random seeds, model parameters, package versions, and a concise run summary.

Raw patient-level records and the raw dataset must not be committed to GitHub. If predictions are saved, include only the minimum fields needed for analysis and do not publish identifying record or hospital keys.

## 8. Acceptance criteria

Part 2 evaluation is ready for model comparison only when:

- [ ] Dataset integrity and modelling-population checks pass.
- [ ] The target is exactly the five classes 1–5.
- [ ] The split audit confirms no hospital overlap in every fold.
- [ ] Class presence and fold-size diagnostics are reviewed and documented.
- [ ] HOSPCODE and prohibited variables are absent from the feature matrix.
- [ ] A fresh preprocessing pipeline is fitted separately within every training fold.
- [ ] The dummy baseline is evaluated on the same folds.
- [ ] Macro-F1 is computed with the fixed five-class label set.
- [ ] Per-class metrics and confusion matrices are generated.
- [ ] Both model implementations use identical saved splits.
- [ ] No unrun, missing, or illustrative result is presented as an observed result.

## 9. Scope and limitations

This design evaluates generalization to held-out hospitals within the 2022 NHAMCS sample. It does not establish clinical safety, causal validity, or prospective performance in a deployed triage system. The reason-for-visit features remain subject to the disclosed Part 1 timing assumption; any sensitivity analysis excluding them must be labelled separately and use the same split design.

**Current status:** plan only. No grouped-CV folds have been audited and no Part 2 classifier metrics have been produced yet.
