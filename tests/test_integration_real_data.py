"""Real-dataset integration check for the Part 2 evaluation framework.

Run explicitly::

    python -m pytest -m integration

Skipped automatically when the pinned NHAMCS file cannot be resolved.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _dataset_available() -> bool:
    try:
        from src import config

        config.resolve_dataset_path()
        return True
    except FileNotFoundError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _dataset_available(),
        reason=(
            "Pinned NHAMCS ed2022-stata.dta not found via "
            "config.resolve_dataset_path(); set NHAMCS_DATA_PATH or place the "
            "file under data/local/."
        ),
    ),
]


@pytest.fixture(scope="module")
def real_inputs():
    from src.data import dataset, target
    from src.evaluation import prepare_modelling_inputs

    full = dataset.load_raw_dataset(verify=True, expect_shape=True)
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)
    return inputs, full


def test_population_and_hospital_counts(real_inputs) -> None:
    inputs, full = real_inputs
    # Revalidate Part 1 audit expectations; fail loudly if the pinned file
    # is the verified 2022 NHAMCS ED file but counts drifted.
    assert len(full) == 16_025
    assert inputs.n_rows == 10_207
    assert inputs.n_hospitals == 147
    assert set(inputs.y.unique()) == {1, 2, 3, 4, 5}


def test_hosPCODE_never_in_feature_matrix(real_inputs) -> None:
    from src.preprocessing.features import FEATURES

    inputs, _ = real_inputs
    assert "HOSPCODE" not in inputs.X.columns
    assert set(inputs.X.columns) == set(FEATURES)


def test_full_evaluation_runs_on_pinned_data(real_inputs) -> None:
    from src.evaluation import run_evaluation

    inputs, _ = real_inputs
    result = run_evaluation(inputs, n_splits=5)

    assert len(result.folds) == 5
    assert len(result.fold_metrics) == 5 * 3 * 2
    assert set(result.fold_metrics["model"]) == {
        "dummy_most_frequent",
        "gaussian_nb",
        "decision_tree_gini",
    }
    assert set(result.fold_metrics["condition"]) == {"full", "no_rfv"}

    for fold in result.folds:
        assert not (fold.train_hospitals & fold.test_hospitals)
        # On the pinned file every test fold should carry all five classes.
        assert all(count > 0 for count in fold.test_class_counts.values())

    for model in result.fold_metrics["model"].unique():
        for condition in result.fold_metrics["condition"].unique():
            subset = result.oof_predictions[
                (result.oof_predictions["model"] == model)
                & (result.oof_predictions["condition"] == condition)
            ]
            assert len(subset) == 10_207
            assert subset["row_index"].is_unique

    dummy_summary = result.summary_for("dummy_most_frequent", "full")
    assert dummy_summary["roc_auc_n_valid"] == 0
    assert np.isnan(dummy_summary["roc_auc_mean"])
    # Majority-class accuracy floor from Part 1 documentation is 0.5232;
    # Dummy most_frequent pooled accuracy should sit at that level.
    pooled_dummy = result.pooled_oof_metrics[
        (result.pooled_oof_metrics["model"] == "dummy_most_frequent")
        & (result.pooled_oof_metrics["condition"] == "full")
    ].iloc[0]
    assert pooled_dummy["accuracy"] == pytest.approx(0.5232, abs=0.001)


def test_part2_baseline_metrics_and_figure_unchanged(real_inputs) -> None:
    import hashlib

    from src.config import FIGURES_DIR
    from src.evaluation import run_evaluation

    inputs, _ = real_inputs
    result = run_evaluation(inputs, n_splits=5)

    tree = result.summary_for("decision_tree_gini", "full")
    assert tree["macro_f1_mean"] == pytest.approx(0.2799, abs=0.0001)

    gaussian = result.summary_for("gaussian_nb", "no_rfv")
    assert gaussian["macro_f1_mean"] == pytest.approx(0.0987, abs=0.0001)
    assert gaussian["macro_f1_std"] == pytest.approx(0.0507, abs=0.0001)

    figure = FIGURES_DIR / "fig05_part2_macro_f1_comparison.png"
    digest = hashlib.sha256(figure.read_bytes()).hexdigest().upper()
    assert digest == "55AAC5DF03B159F4EE9465691E139171C68C70D954A3CAFF944627B013D8833F"


def test_nested_evaluation_runs_on_pinned_data(real_inputs) -> None:
    from src.evaluation import (
        MODEL_FACTORIES,
        NestedSpec,
        run_nested_evaluation,
    )

    inputs, _ = real_inputs
    spec = NestedSpec(
        name="nested_decision_tree_gini",
        estimator_factory=MODEL_FACTORIES["decision_tree_gini"],
        param_grid={"estimator__max_depth": [3, 5]},
    )
    result = run_nested_evaluation(inputs, spec)

    assert len(result.folds) == 5
    assert len(result.fold_results) == 5
    assert len(result.fold_metrics) == 5
    assert len(result.inner_search) == 5 * 3 * 2
    assert len(result.diagnostics) == 5 * 3
    assert result.metadata["total_fits"] == 5 * (3 * 2 + 1)
    assert len(result.oof_predictions) == 10_207
    assert result.oof_predictions["row_index"].is_unique
    assert set(result.oof_predictions["fold"]) == set(range(5))
    assert set(result.oof_predictions["y_pred"]).issubset({1, 2, 3, 4, 5})

    for fold in result.folds:
        assert not (fold.train_hospitals & fold.test_hospitals)
    for fold_result in result.fold_results:
        outer = result.folds[fold_result.fold_id]
        assert len(fold_result.inner_folds) == 3
        for inner in fold_result.inner_folds:
            assert not (inner.train_hospitals & inner.val_hospitals)
            assert inner.val_hospitals.issubset(outer.train_hospitals)
            assert not (inner.val_hospitals & outer.test_hospitals)
