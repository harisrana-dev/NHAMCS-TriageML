"""Unit tests for hospital-grouped splits and reuse."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.evaluation.data import prepare_modelling_inputs
from src.evaluation.runner import MODEL_IDS, run_evaluation
from src.evaluation.splits import (
    GroupedFold,
    fold_diagnostics,
    make_grouped_folds,
    validate_folds,
)


def test_splits_reproducible(synthetic_inputs) -> None:
    folds_a = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)
    folds_b = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)
    assert len(folds_a) == 5
    for a, b in zip(folds_a, folds_b):
        assert np.array_equal(a.train_positions, b.train_positions)
        assert np.array_equal(a.test_positions, b.test_positions)


def test_default_seed_is_project_random_state(synthetic_inputs) -> None:
    folds = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)
    folds_explicit = make_grouped_folds(
        synthetic_inputs.y,
        synthetic_inputs.groups,
        random_state=config.RANDOM_STATE,
    )
    for a, b in zip(folds, folds_explicit):
        assert np.array_equal(a.test_positions, b.test_positions)


def test_no_hospital_overlap(synthetic_inputs) -> None:
    folds = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)
    for fold in folds:
        assert not (fold.train_hospitals & fold.test_hospitals)
        assert set(fold.train_positions).isdisjoint(set(fold.test_positions))


def test_oof_exact_once_coverage(synthetic_inputs) -> None:
    folds = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)
    concatenated = np.concatenate([f.test_positions for f in folds])
    assert concatenated.size == synthetic_inputs.n_rows
    assert np.unique(concatenated).size == synthetic_inputs.n_rows
    assert set(concatenated.tolist()) == set(range(synthetic_inputs.n_rows))


def test_diagnostics_reports_absent_classes(synthetic_inputs) -> None:
    folds = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)
    diagnostics = fold_diagnostics(folds)
    assert list(diagnostics["fold"]) == [0, 1, 2, 3, 4]
    assert (diagnostics["hospital_overlap"] == 0).all()


def test_same_fold_assignments_across_all_six_combinations(synthetic_inputs) -> None:
    """One evaluation run must reuse identical test positions for all pairs."""
    result = run_evaluation(synthetic_inputs, n_splits=5)
    assert len(result.folds) == 5

    # OOF tables for every model-condition share the same fold assignment map.
    oof = result.oof_predictions
    assert set(oof["model"]) == set(MODEL_IDS)
    assert set(oof["condition"]) == {"full", "no_rfv"}
    pivot = oof.pivot_table(
        index="row_index", columns=["model", "condition"], values="fold"
    )
    assert pivot.shape[0] == synthetic_inputs.n_rows
    base = pivot.iloc[:, 0]
    for column in pivot.columns:
        assert np.array_equal(pivot[column].to_numpy(), base.to_numpy()), column


def test_fold_positions_match_result_folds(synthetic_inputs) -> None:
    result = run_evaluation(synthetic_inputs, n_splits=5)
    for fold in result.folds:
        subset = result.oof_predictions[result.oof_predictions["fold"] == fold.fold_id]
        assert set(subset["row_index"]) == set(
            synthetic_inputs.X.index[fold.test_positions]
        )


def test_validate_folds_rejects_duplicate_test_positions() -> None:
    """Regression: the duplicate branch must raise ValueError, not IndexError.

    ``pytest.raises(ValueError)`` fails the test if NumPy's ``IndexError``
    propagates instead of the domain error.
    """
    def make_fold(fold_id: int, train: list[int], test: list[int]) -> GroupedFold:
        return GroupedFold(
            fold_id=fold_id,
            train_positions=np.asarray(train, dtype=np.int64),
            test_positions=np.asarray(test, dtype=np.int64),
            train_hospitals=frozenset({99}),
            test_hospitals=frozenset({fold_id}),
            test_class_counts={1: 0},
        )

    # Four rows, two folds: position 0 lands in both test folds while position
    # 3 is never tested. The concatenated test positions have the correct
    # total size but are not exact-once, so the duplicate branch - not the
    # size check or the per-fold checks - is the one that must fire.
    folds = [
        make_fold(0, train=[2, 3], test=[0, 1]),
        make_fold(1, train=[1, 3], test=[0, 2]),
    ]
    with pytest.raises(ValueError, match="exact-once") as excinfo:
        validate_folds(folds, n_splits=2, n_rows=4)
    assert "[0]" in str(excinfo.value)


def test_validate_folds_accepts_generated_fold_set(synthetic_inputs) -> None:
    folds = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)
    validate_folds(folds, n_splits=len(folds), n_rows=synthetic_inputs.n_rows)
