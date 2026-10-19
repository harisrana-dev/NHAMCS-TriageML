"""Model factories and the shared evaluation runner.

Six model-condition combinations are evaluated on one shared set of hospital-
grouped folds. Each fold fits a **fresh** preprocessing pipeline and estimator
on training rows only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation.data import (
    FEATURE_CONDITIONS,
    LABELS,
    ModellingInputs,
    feature_columns_for,
    select_features,
)
from src.evaluation.metrics import (
    compute_fold_metrics,
    confusion_frame,
    mean_std_summary,
    per_class_frame,
    pooled_metrics_frame,
)
from src.evaluation.splits import GroupedFold, fold_diagnostics, make_grouped_folds
from src.preprocessing import features as ft

#: Stable model identifiers used in result tables.
MODEL_IDS: tuple[str, ...] = (
    "dummy_most_frequent",
    "gaussian_nb",
    "decision_tree_gini",
)

#: Factories return a **new unfitted estimator** on every call.
MODEL_FACTORIES: Mapping[str, Callable[[], Any]] = {
    "dummy_most_frequent": lambda: DummyClassifier(strategy="most_frequent"),
    "gaussian_nb": lambda: GaussianNB(),
    "decision_tree_gini": lambda: DecisionTreeClassifier(
        criterion="gini", random_state=config.RANDOM_STATE
    ),
}

METRIC_COLUMNS: tuple[str, ...] = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
)


def make_estimator(model_id: str):
    """Return a fresh unfitted estimator for ``model_id``."""
    try:
        factory = MODEL_FACTORIES[model_id]
    except KeyError as error:
        known = ", ".join(MODEL_IDS)
        raise KeyError(
            f"Unknown model_id {model_id!r}. Expected one of: {known}."
        ) from error
    return factory()


def build_estimator_pipeline(
    model_id: str,
    columns: Sequence[str] | None = None,
) -> Pipeline:
    """Fresh preprocess + estimator pipeline for one fold/condition.

    Uses :func:`src.preprocessing.features.build_pipeline` so Part 1 and Part 2
    share one ColumnTransformer construction path. The estimator instance is
    always new; never fit a returned pipeline twice.
    """
    estimator = make_estimator(model_id)
    return ft.build_pipeline(estimator=estimator, columns=columns)


@dataclass
class EvaluationResult:
    """Stable result schema for notebooks and tests."""

    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    confusion_matrices: dict[tuple[str, str], pd.DataFrame]
    per_class_metrics: pd.DataFrame
    pooled_oof_metrics: pd.DataFrame
    folds: list[GroupedFold] = field(repr=False)
    diagnostics: pd.DataFrame
    feature_columns: dict[str, tuple[str, ...]]
    fitted_pipelines: dict[tuple[str, str, int], Pipeline] = field(
        default_factory=dict, repr=False
    )

    def summary_for(self, model: str, condition: str) -> pd.Series:
        match = self.summary[
            (self.summary["model"] == model)
            & (self.summary["condition"] == condition)
        ]
        if len(match) != 1:
            raise KeyError(
                f"No unique summary row for model={model!r}, condition={condition!r}"
            )
        return match.iloc[0]


def _fold_metric_row(
    *,
    model: str,
    condition: str,
    fold_id: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "condition": condition,
        "fold": fold_id,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "roc_auc": metrics["roc_auc"],
        "roc_auc_available": metrics["roc_auc_available"],
        "roc_auc_reason": metrics["roc_auc_reason"],
        "n_test": None,
    }
    for label in LABELS:
        block = metrics["per_class"][int(label)]
        row[f"precision_{label}"] = block["precision"]
        row[f"recall_{label}"] = block["recall"]
        row[f"f1_{label}"] = block["f1"]
    return row


def run_evaluation(
    inputs: ModellingInputs,
    *,
    model_ids: Iterable[str] = MODEL_IDS,
    conditions: Iterable[str] | None = None,
    n_splits: int = 5,
    random_state: int | None = None,
    retain_fitted_pipelines: bool = False,
) -> EvaluationResult:
    """Evaluate every model-condition pair on the same grouped folds.

    Splits are generated **once** and reused. Preprocessing and estimators are
    refitted on each training fold only. Set ``retain_fitted_pipelines`` when
    fitted model diagnostics are needed; pipelines are keyed by
    ``(model_id, condition, fold_id)`` and are omitted by default.
    """
    model_ids = tuple(model_ids)
    if conditions is None:
        conditions = tuple(FEATURE_CONDITIONS)
    else:
        conditions = tuple(conditions)
    if not model_ids:
        raise ValueError("At least one model_id is required.")
    if not conditions:
        raise ValueError("At least one feature condition is required.")

    # Validate condition names and column membership up front.
    condition_columns: dict[str, tuple[str, ...]] = {}
    for condition in conditions:
        columns = feature_columns_for(condition)
        ft.leakage_checks(columns)
        condition_columns[condition] = columns

    folds = make_grouped_folds(
        inputs.y,
        inputs.groups,
        n_splits=n_splits,
        random_state=random_state,
    )

    y_all = inputs.y.to_numpy()
    fold_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    confusion: dict[tuple[str, str], pd.DataFrame] = {}
    pooled_rows: list[dict[str, Any]] = []
    fitted_pipelines: dict[tuple[str, str, int], Pipeline] = {}

    for condition in conditions:
        X_condition = select_features(inputs, condition)
        columns = condition_columns[condition]
        for model_id in model_ids:
            oof_pred = np.full(inputs.n_rows, -1, dtype=int)
            oof_fold = np.full(inputs.n_rows, -1, dtype=int)
            oof_true = y_all.copy()

            for fold in folds:
                train_pos = fold.train_positions
                test_pos = fold.test_positions
                X_train = X_condition.iloc[train_pos]
                X_test = X_condition.iloc[test_pos]
                y_train = inputs.y.iloc[train_pos]
                y_test = inputs.y.iloc[test_pos]

                pipeline = build_estimator_pipeline(model_id, columns=columns)
                pipeline.fit(X_train, y_train)
                if retain_fitted_pipelines:
                    fitted_pipelines[(model_id, condition, fold.fold_id)] = pipeline

                y_pred = pipeline.predict(X_test)
                proba_aligned = None
                if hasattr(pipeline, "predict_proba"):
                    # Pipeline.predict_proba transforms with the fold-local
                    # preprocessor, then delegates to the estimator.
                    raw_proba = pipeline.predict_proba(X_test)
                    estimator = pipeline.named_steps["estimator"]
                    proba_aligned = align_via_pipeline(estimator, raw_proba)

                metrics = compute_fold_metrics(
                    y_test.to_numpy(),
                    np.asarray(y_pred),
                    model_id=model_id,
                    proba_aligned=proba_aligned,
                )
                row = _fold_metric_row(
                    model=model_id,
                    condition=condition,
                    fold_id=fold.fold_id,
                    metrics=metrics,
                )
                row["n_test"] = int(test_pos.size)
                fold_rows.append(row)
                per_class_frames.append(
                    per_class_frame(
                        metrics,
                        model=model_id,
                        condition=condition,
                        fold=fold.fold_id,
                    )
                )

                oof_pred[test_pos] = np.asarray(y_pred)
                oof_fold[test_pos] = fold.fold_id

            if np.any(oof_pred < 0):
                missing = int((oof_pred < 0).sum())
                raise RuntimeError(
                    f"Incomplete OOF coverage for {model_id}/{condition}: "
                    f"{missing} row(s) missing a prediction."
                )

            frame = pd.DataFrame(
                {
                    "row_index": inputs.X.index.to_numpy(),
                    "y_true": oof_true,
                    "y_pred": oof_pred,
                    "fold": oof_fold,
                    "model": model_id,
                    "condition": condition,
                }
            )
            oof_frames.append(frame)

            pooled = pooled_metrics_frame(
                oof_true,
                oof_pred,
                model=model_id,
                condition=condition,
            )
            pooled_rows.append(pooled)
            confusion[(model_id, condition)] = confusion_frame(oof_true, oof_pred)

    fold_metrics = pd.DataFrame(fold_rows)
    per_class_metrics = (
        pd.concat(per_class_frames, ignore_index=True)
        if per_class_frames
        else pd.DataFrame()
    )
    oof_predictions = (
        pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    )
    pooled_oof_metrics = pd.DataFrame(pooled_rows)

    # Fold mean/std with explicit sample-std (ddof=1) and n_valid.
    summary_rows: list[dict[str, Any]] = []
    for (model_id, condition), group in fold_metrics.groupby(
        ["model", "condition"], sort=False
    ):
        row: dict[str, Any] = {
            "model": model_id,
            "condition": condition,
            "n_folds": int(len(group)),
        }
        for metric in METRIC_COLUMNS:
            stats = mean_std_summary(group[metric].tolist())
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_n_valid"] = stats["n_valid"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Validate exact-once OOF per model-condition.
    for (model_id, condition), group in oof_predictions.groupby(
        ["model", "condition"], sort=False
    ):
        if len(group) != inputs.n_rows:
            raise RuntimeError(
                f"OOF table for {model_id}/{condition} has {len(group)} rows, "
                f"expected {inputs.n_rows}."
            )
        if group["row_index"].duplicated().any():
            raise RuntimeError(
                f"Duplicate OOF row_index for {model_id}/{condition}."
            )

    return EvaluationResult(
        fold_metrics=fold_metrics,
        summary=summary,
        oof_predictions=oof_predictions,
        confusion_matrices=confusion,
        per_class_metrics=per_class_metrics,
        pooled_oof_metrics=pooled_oof_metrics,
        folds=folds,
        diagnostics=fold_diagnostics(folds),
        feature_columns=condition_columns,
        fitted_pipelines=fitted_pipelines,
    )


def align_via_pipeline(estimator, raw_proba: np.ndarray) -> np.ndarray:
    """Align raw ``predict_proba`` output to the fixed label order."""
    from src.evaluation.metrics import align_proba_columns

    return align_proba_columns(estimator.classes_, raw_proba, LABELS)
