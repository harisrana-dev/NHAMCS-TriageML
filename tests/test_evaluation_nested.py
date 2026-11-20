"""Unit tests for nested hospital-grouped tuning orchestration."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from conftest import synthetic_modelling_frame
from src import config
from src.evaluation.data import LABELS, ModellingInputs, prepare_modelling_inputs
from src.evaluation.nested import (
    NestedSpec,
    make_nested_inner_folds,
    run_nested_evaluation,
)
from src.evaluation.splits import make_grouped_folds
from src.preprocessing.features import NUMERIC


def _spec(param_grid=None, condition="full"):
    grid = (
        {"estimator__max_depth": [1, 2]}
        if param_grid is None
        else param_grid
    )
    return NestedSpec(
        name="nested_tree",
        estimator_factory=lambda: DecisionTreeClassifier(
            random_state=config.RANDOM_STATE
        ),
        param_grid=grid,
        condition=condition,
    )


@pytest.fixture(scope="module")
def nested_inputs():
    return prepare_modelling_inputs(synthetic_modelling_frame())


@pytest.fixture(scope="module")
def nested_result(nested_inputs):
    return run_nested_evaluation(nested_inputs, _spec())


def test_make_nested_inner_folds_are_absolute_and_isolated(nested_inputs) -> None:
    outer_folds = make_grouped_folds(nested_inputs.y, nested_inputs.groups)
    outer = outer_folds[0]
    train_positions = outer.train_positions
    inner_folds = make_nested_inner_folds(
        nested_inputs.y.iloc[train_positions],
        nested_inputs.groups.iloc[train_positions],
        train_positions,
    )

    assert len(inner_folds) == 3
    train_set = set(train_positions.tolist())
    test_set = set(outer.test_positions.tolist())
    for fold in inner_folds:
        assert set(fold.train_positions.tolist()).issubset(train_set)
        assert set(fold.val_positions.tolist()).issubset(train_set)
        assert not set(fold.train_positions.tolist()) & set(fold.val_positions.tolist())
        assert not set(fold.val_positions.tolist()) & test_set
        assert not fold.train_hospitals & fold.val_hospitals
        assert fold.val_hospitals.issubset(outer.train_hospitals)
        assert set(fold.val_class_counts) == set(LABELS)


def test_nested_result_schema_and_exact_once_oof(nested_result, nested_inputs) -> None:
    assert len(nested_result.folds) == 5
    assert len(nested_result.fold_results) == 5
    assert len(nested_result.fold_metrics) == 5
    assert nested_result.fold_metrics["fold"].tolist() == list(range(5))

    oof = nested_result.oof_predictions
    assert len(oof) == nested_inputs.n_rows
    assert oof["row_index"].is_unique
    assert set(oof["y_pred"]).issubset(set(LABELS))
    assert set(oof["fold"]) == set(range(5))

    assert nested_result.best_params["fold"].tolist() == list(range(5))
    assert len(nested_result.inner_search) == 5 * 3 * 2
    assert len(nested_result.diagnostics) == 5 * 3
    assert nested_result.fitted_pipelines == {}
    assert nested_result.confusion_matrix.shape == (5, 5)
    for column in ("macro_f1_mean", "macro_f1_std", "macro_f1_n_valid"):
        assert column in nested_result.summary.columns
    assert nested_result.summary_for("nested_tree", "full")["n_folds"] == 5
    assert nested_result.metadata["total_fits"] == 5 * (3 * 2 + 1)


def test_fitted_pipeline_retention(nested_inputs) -> None:
    result = run_nested_evaluation(
        nested_inputs,
        _spec(),
        retain_fitted_pipelines=True,
    )
    assert set(result.fitted_pipelines) == set(range(5))
    assert all(fold.fitted_pipeline is not None for fold in result.fold_results)

    X = nested_inputs.X.loc[:, list(result.feature_columns)]
    for fold in result.folds:
        predictions = result.fitted_pipelines[fold.fold_id].predict(
            X.iloc[fold.test_positions]
        )
        expected = result.oof_predictions.loc[
            result.oof_predictions["fold"] == fold.fold_id, "y_pred"
        ].to_numpy()
        np.testing.assert_array_equal(predictions, expected)


def test_deterministic_folds_and_first_candidate_tie_break(nested_inputs) -> None:
    grid = {"estimator__random_state": [42, 42]}
    first = run_nested_evaluation(nested_inputs, _spec(grid))
    second = run_nested_evaluation(nested_inputs, _spec(grid))

    pd.testing.assert_frame_equal(first.inner_search, second.inner_search)
    pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)
    pd.testing.assert_frame_equal(first.best_params, second.best_params)
    assert (first.best_params["best_candidate_id"] == 0).all()
    assert (second.best_params["best_candidate_id"] == 0).all()

    for fold_a, fold_b in zip(first.folds, second.folds):
        np.testing.assert_array_equal(fold_a.train_positions, fold_b.train_positions)
        np.testing.assert_array_equal(fold_a.test_positions, fold_b.test_positions)
    for result_a, result_b in zip(first.fold_results, second.fold_results):
        for inner_a, inner_b in zip(result_a.inner_folds, result_b.inner_folds):
            np.testing.assert_array_equal(inner_a.train_positions, inner_b.train_positions)
            np.testing.assert_array_equal(inner_a.val_positions, inner_b.val_positions)


def test_parameter_grid_validation(nested_inputs) -> None:
    with pytest.raises(ValueError, match="Unknown parameter"):
        run_nested_evaluation(nested_inputs, _spec({"estimator__bogus": [1]}))
    with pytest.raises(ValueError, match="non-empty mapping"):
        run_nested_evaluation(nested_inputs, _spec({}))
    with pytest.raises(KeyError, match="Unknown feature condition"):
        run_nested_evaluation(nested_inputs, _spec(condition="dropped_vitals"))
    with pytest.raises(ValueError, match="search_n_jobs"):
        run_nested_evaluation(nested_inputs, _spec(), search_n_jobs=2)


def test_outer_test_perturbation_does_not_change_inner_selection(nested_inputs) -> None:
    folds = make_grouped_folds(nested_inputs.y, nested_inputs.groups)
    fold = folds[0]
    original = run_nested_evaluation(nested_inputs, _spec(), outer_folds=folds)

    perturbed_X = nested_inputs.X.copy()
    perturbed_y = nested_inputs.y.copy()
    test_index = nested_inputs.X.index[fold.test_positions]
    perturbed_X.loc[test_index, list(NUMERIC)] = 999.0
    perturbed_y.iloc[fold.test_positions] = (
        perturbed_y.iloc[fold.test_positions].to_numpy() % 5
    ) + 1
    perturbed_inputs = ModellingInputs(
        X=perturbed_X,
        y=perturbed_y,
        groups=nested_inputs.groups.copy(),
    )

    modified = run_nested_evaluation(perturbed_inputs, _spec(), outer_folds=folds)
    pd.testing.assert_frame_equal(
        original.fold_results[0].inner_search,
        modified.fold_results[0].inner_search,
    )
    assert original.fold_results[0].best_params == modified.fold_results[0].best_params
    original_best = original.best_params.loc[
        original.best_params["fold"] == 0, "best_params"
    ].iloc[0]
    modified_best = modified.best_params.loc[
        modified.best_params["fold"] == 0, "best_params"
    ].iloc[0]
    assert original_best == modified_best


def test_part2_figure_hash_is_unchanged() -> None:
    figure = config.FIGURES_DIR / "fig05_part2_macro_f1_comparison.png"
    digest = hashlib.sha256(figure.read_bytes()).hexdigest().upper()
    assert digest == "55AAC5DF03B159F4EE9465691E139171C68C70D954A3CAFF944627B013D8833F"
