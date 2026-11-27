"""Model factories and tuning specifications for the Part 3 ensembles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite

from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from src import config
from src.evaluation.nested import (
    INVERSE_FREQUENCY_WEIGHT_POLICY,
    NestedSpec,
)

ENSEMBLE_MODEL_IDS: tuple[str, ...] = ("random_forest", "adaboost")

ENSEMBLE_MODEL_FACTORIES: Mapping[str, Callable[[], BaseEstimator]] = {
    "random_forest": lambda: RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    ),
    "adaboost": lambda: AdaBoostClassifier(random_state=config.RANDOM_STATE),
}


def make_ensemble_estimator(model_id: str) -> BaseEstimator:
    """Return a fresh unfitted estimator for one approved ensemble model."""
    try:
        factory = ENSEMBLE_MODEL_FACTORIES[model_id]
    except KeyError as error:
        known = ", ".join(ENSEMBLE_MODEL_IDS)
        raise KeyError(
            f"Unknown ensemble model_id {model_id!r}. Expected one of: {known}."
        ) from error
    return factory()


def make_ensemble_spec(
    model_id: str,
    *,
    condition: str = "full",
) -> NestedSpec:
    """Build the predeclared inner-CV search for one ensemble model."""
    estimator_factory = lambda: make_ensemble_estimator(model_id)
    if model_id == "random_forest":
        return NestedSpec(
            name=model_id,
            estimator_factory=estimator_factory,
            param_grid={
                "estimator__class_weight": [None, "balanced"],
                "estimator__max_depth": [None, 20],
                "estimator__max_features": ["sqrt"],
                "estimator__min_samples_leaf": [1, 5],
                "estimator__n_estimators": [100],
            },
            condition=condition,
        )
    if model_id == "adaboost":
        return NestedSpec(
            name=model_id,
            estimator_factory=estimator_factory,
            param_grid={
                "estimator__learning_rate": [0.5, 1.0],
                "estimator__n_estimators": [50, 100],
            },
            condition=condition,
            weight_policy_candidates=(
                None,
                INVERSE_FREQUENCY_WEIGHT_POLICY,
            ),
        )
    known = ", ".join(ENSEMBLE_MODEL_IDS)
    raise KeyError(
        f"Unknown ensemble model_id {model_id!r}. Expected one of: {known}."
    )


def select_full_condition_winner(
    inner_scores: Mapping[str, list[float]],
) -> str:
    """Choose the family winner by mean best-inner macro-F1, with stable ties."""
    if set(inner_scores) != set(ENSEMBLE_MODEL_IDS):
        raise ValueError(
            "inner_scores must contain exactly: " + ", ".join(ENSEMBLE_MODEL_IDS)
        )
    means: dict[str, float] = {}
    for model_id in ENSEMBLE_MODEL_IDS:
        values = inner_scores[model_id]
        if not values:
            raise ValueError(f"No inner-CV scores supplied for {model_id}.")
        score = sum(float(value) for value in values) / len(values)
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid inner-CV score for {model_id}: {score}.")
        means[model_id] = score
    return max(ENSEMBLE_MODEL_IDS, key=lambda model_id: means[model_id])


__all__ = [
    "ENSEMBLE_MODEL_IDS",
    "ENSEMBLE_MODEL_FACTORIES",
    "make_ensemble_estimator",
    "make_ensemble_spec",
    "select_full_condition_winner",
]
