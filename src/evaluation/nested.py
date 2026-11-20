"""Nested hospital-grouped hyperparameter tuning for Part 3.

This module is additive: Part 2's :func:`src.evaluation.runner.run_evaluation`
and :class:`~src.evaluation.runner.EvaluationResult` are unchanged. Each outer
fold selects parameters with a hospital-grouped inner search fitted only on the
outer-training partition, then scores the held-out outer test fold once.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src import config
from src.evaluation.data import (
    LABELS,
    ModellingInputs,
    feature_columns_for,
    select_features,
)
from src.evaluation.metrics import make_macro_f1_scorer
from src.evaluation.metrics import (
    compute_fold_metrics,
    confusion_frame,
    mean_std_summary,
    per_class_frame,
    pooled_metrics_frame,
)
from src.evaluation.runner import METRIC_COLUMNS, align_via_pipeline
from src.evaluation.splits import (
    GroupedFold,
    fold_diagnostics,
    make_grouped_folds,
    validate_folds,
)
from src.preprocessing import features as ft

DEFAULT_OUTER_SPLITS = 5
DEFAULT_INNER_SPLITS = 3


@dataclass(frozen=True)
class WeightPolicy:
    """Optional sample-weight policy derived only from a fit partition."""

    name: str
    build: Callable[[pd.Series], np.ndarray | None]
    fit_param: str = "sample_weight"


@dataclass(frozen=True)
class NestedSpec:
    """One estimator/condition/grid combination for nested evaluation."""

    name: str
    estimator_factory: Callable[[], BaseEstimator]
    param_grid: Mapping[str, Sequence[Any]]
    condition: str = "full"
    weight_policy: WeightPolicy | None = None


@dataclass(frozen=True)
class NestedInnerFold:
    """One inner split with positions traceable to the full dataset."""

    fold_id: int
    outer_train_positions: np.ndarray
    train_positions: np.ndarray
    val_positions: np.ndarray
    train_hospitals: frozenset
    val_hospitals: frozenset
    val_class_counts: dict[int, int]


@dataclass(frozen=True)
class NestedFoldResult:
    """Selection and outer-test outcome for one outer fold."""

    fold_id: int
    outer_train_positions: np.ndarray
    outer_test_positions: np.ndarray
    inner_folds: tuple[NestedInnerFold, ...]
    best_candidate_id: int
    best_params: dict[str, Any]
    best_inner_score: float
    inner_search: pd.DataFrame
    outer_metrics: dict[str, Any]
    outer_predictions: np.ndarray
    outer_probabilities: np.ndarray | None
    fitted_pipeline: Pipeline | None = None
    weight_policy_name: str = "none"


@dataclass(frozen=True)
class NestedEvaluationResult:
    """Dedicated result schema for nested grouped evaluation."""

    model: str
    condition: str
    folds: tuple[GroupedFold, ...]
    fold_results: tuple[NestedFoldResult, ...]
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    pooled_oof_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    best_params: pd.DataFrame
    inner_search: pd.DataFrame
    diagnostics: pd.DataFrame
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any]
    fitted_pipelines: dict[int, Pipeline] = field(default_factory=dict, repr=False)

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


def inverse_frequency_weights(y: pd.Series) -> np.ndarray:
    """Return ``w_c = n / (K * n_c)`` over the fixed five-class label set."""
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {values.shape}.")
    unknown = sorted(set(np.unique(values).tolist()) - set(LABELS))
    if unknown:
        raise ValueError(f"y contains labels outside {LABELS}: {unknown}")

    n_rows = int(values.size)
    weights = np.zeros(n_rows, dtype=float)
    for label in LABELS:
        mask = values == label
        count = int(mask.sum())
        if count:
            weights[mask] = n_rows / (len(LABELS) * count)
    return weights


INVERSE_FREQUENCY_WEIGHT_POLICY = WeightPolicy(
    name="inverse_frequency",
    build=inverse_frequency_weights,
)


def make_nested_inner_folds(
    y_outer_train: pd.Series,
    groups_outer_train: pd.Series,
    outer_train_positions: np.ndarray,
    *,
    n_splits: int = DEFAULT_INNER_SPLITS,
    random_state: int | None = None,
) -> tuple[NestedInnerFold, ...]:
    """Create grouped inner folds solely from one outer-training partition."""
    if n_splits < 2:
        raise ValueError(f"n_inner_splits must be >= 2, got {n_splits}.")

    positions = _validate_positions(outer_train_positions, "outer_train_positions")
    y_values = np.asarray(y_outer_train)
    group_values = np.asarray(groups_outer_train)
    if y_values.ndim != 1 or group_values.ndim != 1:
        raise ValueError("y_outer_train and groups_outer_train must be 1-D.")
    if len(y_values) != positions.size or len(group_values) != positions.size:
        raise ValueError(
            "Outer-train inputs and positions length mismatch: "
            f"y={len(y_values)}, groups={len(group_values)}, "
            f"positions={positions.size}."
        )

    relative_folds = make_grouped_folds(
        pd.Series(y_values),
        pd.Series(group_values),
        n_splits=n_splits,
        random_state=random_state,
    )

    folds: list[NestedInnerFold] = []
    position_set = set(positions.tolist())
    for relative in relative_folds:
        train_positions = positions[relative.train_positions]
        val_positions = positions[relative.test_positions]
        train_hospitals = frozenset(relative.train_hospitals)
        val_hospitals = frozenset(relative.test_hospitals)

        if train_hospitals & val_hospitals:
            raise ValueError(
                f"Inner fold {relative.fold_id} has hospital overlap between "
                "training and validation."
            )
        if set(train_positions.tolist()) & set(val_positions.tolist()):
            raise ValueError(
                f"Inner fold {relative.fold_id} has position overlap between "
                "training and validation."
            )
        if not set(val_positions.tolist()).issubset(position_set):
            raise ValueError(
                f"Inner fold {relative.fold_id} uses positions outside its "
                "outer-training partition."
            )

        val_class_counts = {
            int(label): int((y_values[relative.test_positions] == label).sum())
            for label in LABELS
        }
        folds.append(
            NestedInnerFold(
                fold_id=relative.fold_id,
                outer_train_positions=positions,
                train_positions=train_positions,
                val_positions=val_positions,
                train_hospitals=train_hospitals,
                val_hospitals=val_hospitals,
                val_class_counts=val_class_counts,
            )
        )
    return tuple(folds)


def build_weighted_fit_params(
    policy: WeightPolicy | None,
    y_fit: pd.Series,
    *,
    final_step_name: str,
) -> dict[str, np.ndarray]:
    """Build estimator-level ``sample_weight`` routing for one fit partition."""
    if policy is None:
        return {}
    if not isinstance(policy, WeightPolicy):
        raise TypeError("weight_policy must be a WeightPolicy or None.")
    if not final_step_name:
        raise ValueError("final_step_name must name the pipeline's final step.")
    if "__" in policy.fit_param:
        raise ValueError(
            "WeightPolicy.fit_param must be an estimator-level parameter name, "
            f"not a pipeline path: {policy.fit_param!r}"
        )

    weights = policy.build(y_fit)
    if weights is None:
        return {}
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.size != len(y_fit):
        raise ValueError(
            f"Weight policy {policy.name!r} returned shape {array.shape}, "
            f"expected ({len(y_fit)},)."
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(
            f"Weight policy {policy.name!r} returned non-finite or negative weights."
        )
    return {f"{final_step_name}__{policy.fit_param}": array}


def run_nested_evaluation(
    inputs: ModellingInputs,
    spec: NestedSpec,
    *,
    outer_folds: Sequence[GroupedFold] | None = None,
    n_outer_splits: int = DEFAULT_OUTER_SPLITS,
    n_inner_splits: int = DEFAULT_INNER_SPLITS,
    random_state: int | None = None,
    scoring: Callable | None = None,
    search_n_jobs: int = 1,
    retain_fitted_pipelines: bool = False,
    error_score: float | str = "raise",
    verbose: int = 0,
) -> NestedEvaluationResult:
    """Select hyperparameters inside each outer fold and score outer OOF."""
    if n_outer_splits < 2:
        raise ValueError(f"n_outer_splits must be >= 2, got {n_outer_splits}.")
    if n_inner_splits < 2:
        raise ValueError(f"n_inner_splits must be >= 2, got {n_inner_splits}.")
    if search_n_jobs != 1:
        raise ValueError(
            "search_n_jobs must be 1; nested evaluation is sequential by protocol."
        )
    error_value = _validate_error_score(error_score)

    columns, base_pipeline, parameter_grid = _validate_spec(spec)
    final_step_name = _pipeline_final_step_name(base_pipeline)
    scorer = scoring if scoring is not None else make_macro_f1_scorer()
    if not callable(scorer):
        raise TypeError("scoring must be callable.")

    if outer_folds is None:
        folds = make_grouped_folds(
            inputs.y,
            inputs.groups,
            n_splits=n_outer_splits,
            random_state=random_state,
        )
    else:
        folds = list(outer_folds)
        validate_folds(folds, n_splits=n_outer_splits, n_rows=inputs.n_rows)

    X_condition = select_features(inputs, spec.condition)
    y_all = inputs.y.to_numpy()
    oof_pred = np.full(inputs.n_rows, -1, dtype=int)
    oof_fold = np.full(inputs.n_rows, -1, dtype=int)

    fold_metric_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    fold_results: list[NestedFoldResult] = []
    best_param_rows: list[dict[str, Any]] = []
    inner_search_frames: list[pd.DataFrame] = []
    fitted_pipelines: dict[int, Pipeline] = {}

    n_candidates = len(parameter_grid)
    if n_candidates == 0:
        raise ValueError("parameter_grid produced no candidates.")

    for fold in folds:
        outer_train_pos = _validate_positions(
            fold.train_positions, "fold.train_positions"
        )
        outer_test_pos = _validate_positions(
            fold.test_positions, "fold.test_positions"
        )
        if fold.train_hospitals & fold.test_hospitals:
            raise ValueError(
                f"Outer fold {fold.fold_id} has hospital overlap between "
                "training and test."
            )
        if set(outer_train_pos.tolist()) & set(outer_test_pos.tolist()):
            raise ValueError(
                f"Outer fold {fold.fold_id} has position overlap between "
                "training and test."
            )

        inner_folds = make_nested_inner_folds(
            inputs.y.iloc[outer_train_pos],
            inputs.groups.iloc[outer_train_pos],
            outer_train_pos,
            n_splits=n_inner_splits,
            random_state=random_state,
        )
        for inner in inner_folds:
            if not set(inner.val_positions.tolist()).issubset(
                set(outer_train_pos.tolist())
            ):
                raise ValueError(
                    f"Outer fold {fold.fold_id}: inner validation positions "
                    "escape the outer-training partition."
                )
            if set(inner.val_positions.tolist()) & set(outer_test_pos.tolist()):
                raise ValueError(
                    f"Outer fold {fold.fold_id}: inner validation uses outer "
                    "test positions."
                )
            if not set(inner.val_hospitals).issubset(set(fold.train_hospitals)):
                raise ValueError(
                    f"Outer fold {fold.fold_id}: inner validation hospitals "
                    "include an outer test hospital."
                )

        candidate_summary: list[dict[str, Any]] = []
        candidate_rows: list[list[dict[str, Any]]] = []
        for candidate_id, params in enumerate(parameter_grid):
            scores: list[float] = []
            rows_for_candidate: list[dict[str, Any]] = []
            for inner in inner_folds:
                score = error_value
                failed = False
                error_message = ""
                try:
                    pipeline = clone(base_pipeline).set_params(**params)
                    fit_y = inputs.y.iloc[inner.train_positions]
                    fit_params = build_weighted_fit_params(
                        spec.weight_policy,
                        fit_y,
                        final_step_name=final_step_name,
                    )
                    pipeline.fit(
                        X_condition.iloc[inner.train_positions],
                        fit_y,
                        **fit_params,
                    )
                    score = float(
                        scorer(
                            pipeline,
                            X_condition.iloc[inner.val_positions],
                            inputs.y.iloc[inner.val_positions],
                        )
                    )
                except Exception as error:
                    if error_value == "raise":
                        raise RuntimeError(
                            f"Inner fit failed for outer fold {fold.fold_id}, "
                            f"inner fold {inner.fold_id}, candidate {candidate_id}."
                        ) from error
                    failed = True
                    error_message = f"{type(error).__name__}: {error}"
                scores.append(score)
                rows_for_candidate.append(
                    {
                        "model": spec.name,
                        "condition": spec.condition,
                        "outer_fold": fold.fold_id,
                        "candidate_id": candidate_id,
                        "params": _params_json(params),
                        "inner_fold": inner.fold_id,
                        "n_inner_train": int(inner.train_positions.size),
                        "n_inner_val": int(inner.val_positions.size),
                        "score": score,
                        "mean_score": float("nan"),
                        "failed": failed,
                        "error": error_message,
                    }
                )
            mean_score = float(np.mean(scores))
            for row in rows_for_candidate:
                row["mean_score"] = mean_score
            candidate_rows.append(rows_for_candidate)
            candidate_summary.append(
                {
                    "candidate_id": candidate_id,
                    "params": dict(params),
                    "mean_score": mean_score,
                }
            )

        best_index = int(
            np.argmax([candidate["mean_score"] for candidate in candidate_summary])
        )
        best = candidate_summary[best_index]
        best_params = dict(best["params"])
        best_score = float(best["mean_score"])

        if verbose:
            print(
                f"outer_fold={fold.fold_id} best_candidate={best['candidate_id']} "
                f"best_inner_score={best_score:.6f}"
            )

        final_pipeline = clone(base_pipeline).set_params(**best_params)
        outer_fit_y = inputs.y.iloc[outer_train_pos]
        final_fit_params = build_weighted_fit_params(
            spec.weight_policy,
            outer_fit_y,
            final_step_name=final_step_name,
        )
        final_pipeline.fit(
            X_condition.iloc[outer_train_pos],
            outer_fit_y,
            **final_fit_params,
        )

        y_pred = np.asarray(final_pipeline.predict(X_condition.iloc[outer_test_pos]))
        proba_aligned = None
        if hasattr(final_pipeline, "predict_proba"):
            raw_proba = final_pipeline.predict_proba(X_condition.iloc[outer_test_pos])
            estimator = final_pipeline.named_steps[final_step_name]
            proba_aligned = align_via_pipeline(estimator, raw_proba)

        outer_metrics = compute_fold_metrics(
            y_all[outer_test_pos],
            y_pred,
            model_id=spec.name,
            proba_aligned=proba_aligned,
        )
        metric_row = _nested_fold_metric_row(
            model=spec.name,
            condition=spec.condition,
            fold_id=fold.fold_id,
            metrics=outer_metrics,
            n_test=int(outer_test_pos.size),
        )
        fold_metric_rows.append(metric_row)
        per_class_frames.append(
            per_class_frame(
                outer_metrics,
                model=spec.name,
                condition=spec.condition,
                fold=fold.fold_id,
            )
        )
        oof_pred[outer_test_pos] = y_pred
        oof_fold[outer_test_pos] = fold.fold_id

        retained = final_pipeline if retain_fitted_pipelines else None
        if retain_fitted_pipelines:
            fitted_pipelines[fold.fold_id] = final_pipeline
        fold_results.append(
            NestedFoldResult(
                fold_id=fold.fold_id,
                outer_train_positions=outer_train_pos,
                outer_test_positions=outer_test_pos,
                inner_folds=inner_folds,
                best_candidate_id=int(best["candidate_id"]),
                best_params=best_params,
                best_inner_score=best_score,
                inner_search=pd.DataFrame(
                    [row for rows in candidate_rows for row in rows]
                ),
                outer_metrics=outer_metrics,
                outer_predictions=y_pred,
                outer_probabilities=proba_aligned,
                fitted_pipeline=retained,
                weight_policy_name=(
                    spec.weight_policy.name if spec.weight_policy is not None else "none"
                ),
            )
        )

        best_row: dict[str, Any] = {
            "model": spec.name,
            "condition": spec.condition,
            "fold": fold.fold_id,
            "best_candidate_id": int(best["candidate_id"]),
            "best_inner_score": best_score,
            "best_params": _params_json(best_params),
            "weight_policy": (
                spec.weight_policy.name if spec.weight_policy is not None else "none"
            ),
        }
        for key, value in best_params.items():
            best_row[f"param__{key}"] = value
        best_param_rows.append(best_row)

    if np.any(oof_pred < 0):
        missing = int((oof_pred < 0).sum())
        raise RuntimeError(f"Incomplete OOF coverage: {missing} row(s) missing.")

    oof_predictions = pd.DataFrame(
        {
            "row_index": inputs.X.index.to_numpy(),
            "y_true": np.asarray(y_all).copy(),
            "y_pred": oof_pred,
            "fold": oof_fold,
            "model": spec.name,
            "condition": spec.condition,
        }
    )
    if len(oof_predictions) != inputs.n_rows:
        raise RuntimeError(
            f"OOF table has {len(oof_predictions)} rows, expected {inputs.n_rows}."
        )
    if oof_predictions["row_index"].duplicated().any():
        raise RuntimeError("OOF table contains duplicate row_index values.")

    fold_metrics = pd.DataFrame(fold_metric_rows)
    inner_search = pd.concat(
        [result.inner_search for result in fold_results], ignore_index=True
    )
    summary = _summary_frame(fold_metrics, spec.name, spec.condition)
    pooled = pooled_metrics_frame(
        y_all,
        oof_pred,
        model=spec.name,
        condition=spec.condition,
    )

    metadata = {
        "model": spec.name,
        "condition": spec.condition,
        "random_state": (
            config.RANDOM_STATE if random_state is None else int(random_state)
        ),
        "n_outer_splits": int(n_outer_splits),
        "n_inner_splits": int(n_inner_splits),
        "n_candidates": int(n_candidates),
        "total_fits": int(len(folds) * (n_inner_splits * n_candidates + 1)),
        "search_n_jobs": int(search_n_jobs),
        "scoring": getattr(scorer, "__name__", type(scorer).__name__),
        "error_score": error_score if error_value == "raise" else error_value,
        "weight_policy": (
            spec.weight_policy.name if spec.weight_policy is not None else "none"
        ),
        "group_column": "HOSPCODE",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }

    return NestedEvaluationResult(
        model=spec.name,
        condition=spec.condition,
        folds=tuple(folds),
        fold_results=tuple(fold_results),
        fold_metrics=fold_metrics,
        summary=summary,
        oof_predictions=oof_predictions,
        pooled_oof_metrics=pd.DataFrame([pooled]),
        per_class_metrics=(
            pd.concat(per_class_frames, ignore_index=True)
            if per_class_frames
            else pd.DataFrame()
        ),
        confusion_matrix=confusion_frame(y_all, oof_pred),
        best_params=pd.DataFrame(best_param_rows),
        inner_search=inner_search,
        diagnostics=_nested_diagnostics(fold_results),
        feature_columns=columns,
        metadata=metadata,
        fitted_pipelines=fitted_pipelines,
    )


__all__ = [
    "INVERSE_FREQUENCY_WEIGHT_POLICY",
    "NestedEvaluationResult",
    "NestedFoldResult",
    "NestedInnerFold",
    "NestedSpec",
    "WeightPolicy",
    "build_weighted_fit_params",
    "inverse_frequency_weights",
    "make_nested_inner_folds",
    "run_nested_evaluation",
]


def _validate_positions(positions: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(positions, dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D index array.")
    if np.any(array < 0):
        raise ValueError(f"{name} contains negative positions.")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} contains duplicate positions.")
    return array


def _validate_error_score(error_score: float | str) -> float | str:
    if isinstance(error_score, str):
        if error_score != "raise":
            raise ValueError("error_score strings other than 'raise' are not supported.")
        return error_score
    value = float(error_score)
    if not np.isfinite(value):
        raise ValueError("Numeric error_score must be finite.")
    return value


def _pipeline_final_step_name(pipeline: Pipeline) -> str:
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Expected a sklearn Pipeline.")
    name = pipeline.steps[-1][0]
    if not name:
        raise ValueError("Pipeline final step has no name.")
    return name


def _validate_spec(
    spec: NestedSpec,
) -> tuple[tuple[str, ...], Pipeline, ParameterGrid]:
    if not isinstance(spec, NestedSpec):
        raise TypeError("spec must be a NestedSpec.")
    if not spec.name:
        raise ValueError("NestedSpec.name must be non-empty.")
    estimator = spec.estimator_factory()
    if not isinstance(estimator, BaseEstimator):
        raise TypeError("estimator_factory must return a sklearn BaseEstimator.")

    columns = feature_columns_for(spec.condition)
    ft.leakage_checks(columns)
    if not isinstance(spec.param_grid, Mapping) or not spec.param_grid:
        raise ValueError("param_grid must be a non-empty mapping.")

    base_pipeline = ft.build_pipeline(estimator=estimator, columns=columns)
    valid_params = set(base_pipeline.get_params(deep=True))
    for key in spec.param_grid:
        if not isinstance(key, str) or not key:
            raise ValueError("Parameter grid keys must be non-empty strings.")
        if key not in valid_params:
            raise ValueError(
                f"Unknown parameter {key!r} for {type(estimator).__name__} pipeline. "
                "Use pipeline-prefixed names such as 'estimator__max_depth'."
            )
    return columns, base_pipeline, ParameterGrid(spec.param_grid)


def _params_json(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True, default=str)


def _nested_fold_metric_row(
    *,
    model: str,
    condition: str,
    fold_id: int,
    metrics: dict[str, Any],
    n_test: int,
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
        "n_test": n_test,
    }
    for label in LABELS:
        block = metrics["per_class"][int(label)]
        row[f"precision_{label}"] = block["precision"]
        row[f"recall_{label}"] = block["recall"]
        row[f"f1_{label}"] = block["f1"]
    return row


def _summary_frame(
    fold_metrics: pd.DataFrame,
    model: str,
    condition: str,
) -> pd.DataFrame:
    row: dict[str, Any] = {
        "model": model,
        "condition": condition,
        "n_folds": int(len(fold_metrics)),
    }
    for metric in METRIC_COLUMNS:
        stats = mean_std_summary(fold_metrics[metric].tolist())
        row[f"{metric}_mean"] = stats["mean"]
        row[f"{metric}_std"] = stats["std"]
        row[f"{metric}_n_valid"] = stats["n_valid"]
    return pd.DataFrame([row])


def _nested_diagnostics(fold_results: Sequence[NestedFoldResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in fold_results:
        for inner in result.inner_folds:
            rows.append(
                {
                    "outer_fold": result.fold_id,
                    "inner_fold": inner.fold_id,
                    "n_outer_train": int(result.outer_train_positions.size),
                    "n_outer_test": int(result.outer_test_positions.size),
                    "n_inner_train": int(inner.train_positions.size),
                    "n_inner_val": int(inner.val_positions.size),
                    "n_train_hospitals": len(inner.train_hospitals),
                    "n_val_hospitals": len(inner.val_hospitals),
                    "hospital_overlap": len(
                        inner.train_hospitals & inner.val_hospitals
                    ),
                    "val_class_counts": inner.val_class_counts,
                    "best_candidate_id": result.best_candidate_id,
                    "weight_policy": result.weight_policy_name,
                }
            )
    return pd.DataFrame(rows)
