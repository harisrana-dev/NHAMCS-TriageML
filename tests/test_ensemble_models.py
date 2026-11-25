"""Tests for the Part 3 ensemble factories and inner-CV specifications."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from src import config
from src.evaluation import (
    ENSEMBLE_MODEL_IDS,
    INVERSE_FREQUENCY_WEIGHT_POLICY,
    make_ensemble_estimator,
    make_ensemble_spec,
    run_nested_evaluation,
    select_full_condition_winner,
)


def test_ensemble_factories_are_fresh_seeded_and_single_threaded() -> None:
    assert ENSEMBLE_MODEL_IDS == ("random_forest", "adaboost")
    rf_a = make_ensemble_estimator("random_forest")
    rf_b = make_ensemble_estimator("random_forest")
    ada_a = make_ensemble_estimator("adaboost")
    ada_b = make_ensemble_estimator("adaboost")

    assert isinstance(rf_a, RandomForestClassifier)
    assert isinstance(ada_a, AdaBoostClassifier)
    assert rf_a is not rf_b
    assert ada_a is not ada_b
    assert rf_a.random_state == config.RANDOM_STATE
    assert rf_a.n_jobs == 1
    assert ada_a.random_state == config.RANDOM_STATE


def test_weighting_strategies_are_part_of_inner_search() -> None:
    rf_spec = make_ensemble_spec("random_forest")
    ada_spec = make_ensemble_spec("adaboost")

    assert rf_spec.param_grid["estimator__class_weight"] == [None, "balanced"]
    assert rf_spec.weight_policy_candidates is None
    assert ada_spec.weight_policy_candidates == (
        None,
        INVERSE_FREQUENCY_WEIGHT_POLICY,
    )
    assert ada_spec.condition == "full"
    assert make_ensemble_spec("adaboost", condition="no_rfv").condition == "no_rfv"

    with pytest.raises(KeyError, match="Unknown ensemble"):
        make_ensemble_estimator("gradient_boosting")


def test_ada_boost_nested_search_records_inner_selected_weighting(
    synthetic_inputs,
) -> None:
    spec = make_ensemble_spec("adaboost")
    small_spec = replace(
        spec,
        param_grid={
            "estimator__learning_rate": [1.0],
            "estimator__n_estimators": [5],
        },
    )
    result = run_nested_evaluation(synthetic_inputs, small_spec)

    assert len(result.inner_search) == 5 * 3 * 2
    assert set(result.inner_search["weight_policy"]) == {
        "none",
        "inverse_frequency",
    }
    assert set(result.best_params["weight_policy"]).issubset(
        {"none", "inverse_frequency"}
    )
    assert result.metadata["weight_policy_candidates"] == [
        "none",
        "inverse_frequency",
    ]


def test_random_forest_nested_search_compares_native_class_weight(
    synthetic_inputs,
) -> None:
    spec = make_ensemble_spec("random_forest")
    small_spec = replace(
        spec,
        param_grid={
            "estimator__class_weight": [None, "balanced"],
            "estimator__max_depth": [3],
            "estimator__max_features": ["sqrt"],
            "estimator__min_samples_leaf": [2],
            "estimator__n_estimators": [5],
        },
    )
    result = run_nested_evaluation(synthetic_inputs, small_spec)

    assert len(result.inner_search) == 5 * 3 * 2
    candidates = set(result.inner_search["params"].map(json.loads).map(
        lambda params: params["estimator__class_weight"]
    ))
    assert candidates == {None, "balanced"}
    assert set(result.best_params["weight_policy"]) == {"none"}


def test_family_winner_uses_only_precomputed_inner_scores() -> None:
    assert select_full_condition_winner(
        {"random_forest": [0.20, 0.30], "adaboost": [0.25, 0.26]}
    ) == "adaboost"
    assert select_full_condition_winner(
        {"random_forest": [0.3], "adaboost": [0.3]}
    ) == "random_forest"

    with pytest.raises(ValueError, match="exactly"):
        select_full_condition_winner({"random_forest": [0.3]})
    with pytest.raises(ValueError, match="Invalid"):
        select_full_condition_winner(
            {"random_forest": [float("nan")], "adaboost": [0.2]}
        )
