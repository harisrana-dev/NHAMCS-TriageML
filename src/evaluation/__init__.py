"""Hospital-grouped cross-validation and Part 2 evaluation metrics.

Issue #1 delivers the shared evaluation layer only. Model interpretation,
notebook integration, and the coursework comparison belong to later issues.

Typical usage::

    from src.data import dataset, target
    from src.evaluation import prepare_modelling_inputs, run_evaluation

    full = dataset.load_raw_dataset()
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)
    result = run_evaluation(inputs)

    result.summary
    result.fold_metrics
    result.oof_predictions
"""

from src.evaluation.data import (
    FEATURE_CONDITIONS,
    GROUP_COLUMN,
    LABELS,
    RFV_COLUMNS,
    ModellingInputs,
    feature_columns_for,
    prepare_modelling_inputs,
    select_features,
)
from src.evaluation.metrics import (
    ROC_AUC_PROTOCOL_NA,
    align_proba_columns,
    compute_fold_metrics,
    confusion_frame,
    fold_roc_auc,
    make_macro_f1_scorer,
    mean_std_summary,
)
from src.evaluation.runner import (
    MODEL_FACTORIES,
    MODEL_IDS,
    EvaluationResult,
    build_estimator_pipeline,
    make_estimator,
    run_evaluation,
)
from src.evaluation.nested import (
    INVERSE_FREQUENCY_WEIGHT_POLICY,
    NestedEvaluationResult,
    NestedFoldResult,
    NestedInnerFold,
    NestedSpec,
    WeightPolicy,
    build_weighted_fit_params,
    inverse_frequency_weights,
    make_nested_inner_folds,
    run_nested_evaluation,
)
from src.evaluation.splits import (
    GroupedFold,
    fold_diagnostics,
    make_grouped_folds,
    validate_folds,
)

__all__ = [
    "LABELS",
    "RFV_COLUMNS",
    "FEATURE_CONDITIONS",
    "GROUP_COLUMN",
    "ModellingInputs",
    "prepare_modelling_inputs",
    "feature_columns_for",
    "select_features",
    "GroupedFold",
    "make_grouped_folds",
    "validate_folds",
    "fold_diagnostics",
    "MODEL_IDS",
    "MODEL_FACTORIES",
    "make_estimator",
    "build_estimator_pipeline",
    "run_evaluation",
    "EvaluationResult",
    "align_proba_columns",
    "compute_fold_metrics",
    "fold_roc_auc",
    "confusion_frame",
    "mean_std_summary",
    "ROC_AUC_PROTOCOL_NA",
    "make_macro_f1_scorer",
    "NestedEvaluationResult",
    "NestedFoldResult",
    "NestedInnerFold",
    "NestedSpec",
    "WeightPolicy",
    "INVERSE_FREQUENCY_WEIGHT_POLICY",
    "inverse_frequency_weights",
    "make_nested_inner_folds",
    "build_weighted_fit_params",
    "run_nested_evaluation",
]
