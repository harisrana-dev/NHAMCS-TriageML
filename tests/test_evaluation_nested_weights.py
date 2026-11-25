"""Unit tests for nested-evaluation sample-weight policies."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation.nested import (
    INVERSE_FREQUENCY_WEIGHT_POLICY,
    NestedSpec,
    WeightPolicy,
    build_weighted_fit_params,
    inverse_frequency_weights,
    run_nested_evaluation,
)

FIT_RECORDS: list[tuple[pd.Series, np.ndarray | None]] = []


class RecordingClassifier(BaseEstimator, ClassifierMixin):
    """Fixed-label classifier that records the exact sample weights it receives."""

    def __init__(self, label: int = 1):
        self.label = label

    def fit(self, X, y, sample_weight=None):
        FIT_RECORDS.append((pd.Series(np.asarray(y)), sample_weight))
        self.classes_ = np.asarray([1, 2, 3, 4, 5], dtype=int)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.label, dtype=int)


def _recording_spec(grid=None, policy=None):
    return NestedSpec(
        name="recording",
        estimator_factory=RecordingClassifier,
        param_grid=grid or {"estimator__label": [1, 3]},
        weight_policy=policy,
    )


def test_inverse_frequency_weights_exact_arithmetic() -> None:
    y = pd.Series([1, 1, 1, 2, 2, 3])
    weights = inverse_frequency_weights(y)

    expected = np.array([6 / 15] * 3 + [6 / 10] * 2 + [6 / 5])
    np.testing.assert_allclose(weights, expected)
    assert np.isclose(weights.sum(), len(y) * 3 / 5)

    with pytest.raises(ValueError, match="outside"):
        inverse_frequency_weights(pd.Series([1, 9]))


def test_build_weighted_fit_params_routes_to_final_estimator() -> None:
    y = pd.Series([1, 1, 1, 2, 2])
    params = build_weighted_fit_params(
        INVERSE_FREQUENCY_WEIGHT_POLICY,
        y,
        final_step_name="estimator",
    )

    assert list(params) == ["estimator__sample_weight"]
    np.testing.assert_allclose(
        params["estimator__sample_weight"],
        inverse_frequency_weights(y),
    )

    bad_path = WeightPolicy(
        name="bad_path",
        build=lambda values: np.ones(len(values)),
        fit_param="estimator__sample_weight",
    )
    with pytest.raises(ValueError, match="pipeline path"):
        build_weighted_fit_params(bad_path, y, final_step_name="estimator")

    bad_shape = WeightPolicy(
        name="bad_shape",
        build=lambda values: np.ones(len(values) + 1),
    )
    with pytest.raises(ValueError, match="shape"):
        build_weighted_fit_params(bad_shape, y, final_step_name="estimator")

    assert build_weighted_fit_params(None, y, final_step_name="estimator") == {}


def test_sample_weights_recomputed_per_partition(synthetic_inputs) -> None:
    FIT_RECORDS.clear()
    result = run_nested_evaluation(
        synthetic_inputs,
        _recording_spec(policy=INVERSE_FREQUENCY_WEIGHT_POLICY),
    )

    assert len(FIT_RECORDS) == 5 * (3 * 2 + 1)
    for y_used, weights in FIT_RECORDS:
        assert weights is not None
        np.testing.assert_allclose(
            weights,
            inverse_frequency_weights(y_used),
        )
    assert set(result.best_params["weight_policy"]) == {"inverse_frequency"}
    assert set(result.diagnostics["weight_policy"]) == {"inverse_frequency"}


def test_without_policy_no_sample_weights_are_passed(synthetic_inputs) -> None:
    FIT_RECORDS.clear()
    result = run_nested_evaluation(
        synthetic_inputs,
        _recording_spec(grid={"estimator__label": [1]}),
    )

    assert len(FIT_RECORDS) == 5 * (3 * 1 + 1)
    assert all(weights is None for _, weights in FIT_RECORDS)
    assert set(result.best_params["weight_policy"]) == {"none"}


def test_weight_policies_are_compared_inside_inner_cv(synthetic_inputs) -> None:
    FIT_RECORDS.clear()
    result = run_nested_evaluation(
        synthetic_inputs,
        NestedSpec(
            name="recording",
            estimator_factory=RecordingClassifier,
            param_grid={"estimator__label": [1]},
            weight_policy_candidates=(
                INVERSE_FREQUENCY_WEIGHT_POLICY,
                None,
            ),
        ),
    )

    assert len(result.inner_search) == 5 * 3 * 2
    assert set(result.inner_search["weight_policy"]) == {
        "inverse_frequency",
        "none",
    }
    assert set(result.best_params["weight_policy"]) == {"inverse_frequency"}
    assert all(
        fold.weight_policy_name == "inverse_frequency"
        for fold in result.fold_results
    )
    assert len(FIT_RECORDS) == 5 * (3 * 2 + 1)
    assert sum(weights is not None for _, weights in FIT_RECORDS) == 5 * (3 + 1)
    for y_fit, weights in FIT_RECORDS:
        if weights is not None:
            np.testing.assert_allclose(weights, inverse_frequency_weights(y_fit))


def test_estimator_native_class_weight_is_an_ordinary_grid_param(
    synthetic_inputs,
) -> None:
    spec = NestedSpec(
        name="tree_class_weight",
        estimator_factory=lambda: DecisionTreeClassifier(
            random_state=config.RANDOM_STATE
        ),
        param_grid={"estimator__class_weight": [None, "balanced"]},
    )
    result = run_nested_evaluation(synthetic_inputs, spec)

    assert len(result.inner_search) == 5 * 3 * 2
    assert all(
        "estimator__class_weight" in json.loads(row)
        for row in result.inner_search["params"]
    )
