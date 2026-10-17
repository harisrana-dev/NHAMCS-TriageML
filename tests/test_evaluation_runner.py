"""Unit tests for model factories, fresh fitting, and the evaluation runner."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.evaluation.data import FEATURE_CONDITIONS, LABELS, select_features
from src.evaluation.metrics import ROC_AUC_PROTOCOL_NA
from src.evaluation.runner import (
    MODEL_FACTORIES,
    MODEL_IDS,
    build_estimator_pipeline,
    make_estimator,
    run_evaluation,
)
from src.preprocessing import features as ft


def test_model_factories_return_fresh_instances() -> None:
    for model_id in MODEL_IDS:
        a = make_estimator(model_id)
        b = make_estimator(model_id)
        assert a is not b
        assert clone(a) is not a


def test_unknown_model_id_fails() -> None:
    with pytest.raises(KeyError, match="Unknown model_id"):
        make_estimator("random_forest")


def test_pipeline_uses_shared_preprocess_and_estimator() -> None:
    pipeline = build_estimator_pipeline("decision_tree_gini")
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess", "estimator"]
    assert not isinstance(pipeline.named_steps["estimator"], str)
    # Same ColumnTransformer construction path as Part 1 (unfitted view).
    block_names = [name for name, _, _ in pipeline.named_steps["preprocess"].transformers]
    assert "categorical_high" in block_names


def test_no_rfv_pipeline_drops_rfv_block_columns() -> None:
    columns = FEATURE_CONDITIONS["no_rfv"]
    pipeline = build_estimator_pipeline("gaussian_nb", columns=columns)
    ct = pipeline.named_steps["preprocess"]
    matched = [item for item in ct.transformers if item[0] == "categorical_high"]
    assert matched
    high_columns = list(matched[0][2])
    assert high_columns == []
    assert set(columns).isdisjoint(set(ft.CATEGORICAL_HIGH))
    # Low-cardinality and numeric blocks still present
    low = [item for item in ct.transformers if item[0] == "categorical_low"][0]
    assert list(low[2]) == list(ft.CATEGORICAL_LOW)


def test_every_fold_gets_fresh_pipeline(synthetic_inputs) -> None:
    seen = []
    original = build_estimator_pipeline

    def spy(model_id, columns=None):
        pipeline = original(model_id, columns=columns)
        seen.append(pipeline)
        return pipeline

    import src.evaluation.runner as runner

    runner_build = runner.build_estimator_pipeline
    runner.build_estimator_pipeline = spy
    try:
        result = run_evaluation(
            synthetic_inputs,
            model_ids=("dummy_most_frequent",),
            conditions=("full",),
            n_splits=5,
        )
    finally:
        runner.build_estimator_pipeline = runner_build

    assert len(seen) == 5
    assert len({id(p) for p in seen}) == 5
    assert len(result.folds) == 5


def test_preprocess_fitted_only_on_training_fold(synthetic_inputs) -> None:
    """Imputer statistics must match a train-only fit, not a full-data fit."""
    fold = run_evaluation(
        synthetic_inputs,
        model_ids=("dummy_most_frequent",),
        conditions=("full",),
        n_splits=5,
    ).folds[0]
    X = synthetic_inputs.X
    columns = FEATURE_CONDITIONS["full"]
    train_pos = fold.train_positions

    pipeline = build_estimator_pipeline("dummy_most_frequent", columns=columns)
    pipeline.fit(X.iloc[train_pos], synthetic_inputs.y.iloc[train_pos])
    imputer = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
    )
    numeric_values = X.iloc[train_pos][list(ft.NUMERIC)].to_numpy(dtype=float)
    medians = np.nanmedian(numeric_values, axis=0)
    np.testing.assert_allclose(imputer.statistics_, medians, rtol=1e-6)

    full_medians = np.nanmedian(X[list(ft.NUMERIC)].to_numpy(dtype=float), axis=0)
    if np.allclose(medians, full_medians):
        assert np.allclose(imputer.statistics_, medians)
    else:
        assert not np.allclose(imputer.statistics_, full_medians)


def test_result_schema_and_exact_once_oof(synthetic_inputs) -> None:
    result = run_evaluation(
        synthetic_inputs,
        model_ids=MODEL_IDS,
        conditions=("full", "no_rfv"),
        n_splits=5,
    )
    expected_pairs = {(m, c) for m in MODEL_IDS for c in ("full", "no_rfv")}
    pairs = set(zip(result.fold_metrics["model"], result.fold_metrics["condition"]))
    assert pairs == expected_pairs
    assert len(result.fold_metrics) == 5 * 6

    assert set(result.summary["model"]) == {m for m, _ in expected_pairs}
    for metric in ("macro_f1", "accuracy", "balanced_accuracy", "roc_auc"):
        assert f"{metric}_mean" in result.summary.columns
        assert f"{metric}_std" in result.summary.columns
        assert f"{metric}_n_valid" in result.summary.columns

    oof = result.oof_predictions
    for model, condition in expected_pairs:
        subset = oof[(oof["model"] == model) & (oof["condition"] == condition)]
        assert len(subset) == synthetic_inputs.n_rows
        assert subset["row_index"].is_unique
        assert set(subset["y_pred"]).issubset(set(LABELS))
        assert set(subset["y_true"]).issubset(set(LABELS))

    # Pooled OOF metrics present for every combination
    pooled_pairs = set(
        zip(result.pooled_oof_metrics["model"], result.pooled_oof_metrics["condition"])
    )
    assert pooled_pairs == expected_pairs

    # Confusion matrices
    assert set(result.confusion_matrices) == expected_pairs
    for frame in result.confusion_matrices.values():
        assert frame.shape == (5, 5)


def test_dummy_roc_auc_n_valid_is_zero(synthetic_inputs) -> None:
    result = run_evaluation(
        synthetic_inputs,
        model_ids=("dummy_most_frequent",),
        conditions=("full",),
        n_splits=5,
    )
    summary = result.summary_for("dummy_most_frequent", "full")
    assert summary["roc_auc_n_valid"] == 0
    assert np.isnan(summary["roc_auc_mean"])
    reasons = result.fold_metrics["roc_auc_reason"].unique().tolist()
    assert reasons == [ROC_AUC_PROTOCOL_NA]


def test_fold_metrics_probability_alignment_smoke(synthetic_inputs) -> None:
    """Tree/GNB folds should produce finite or explicitly invalid AUC."""
    result = run_evaluation(
        synthetic_inputs,
        model_ids=("gaussian_nb", "decision_tree_gini"),
        conditions=("full",),
        n_splits=5,
    )
    for _, row in result.fold_metrics.iterrows():
        if row["roc_auc_available"]:
            assert np.isfinite(row["roc_auc"])
        else:
            assert isinstance(row["roc_auc_reason"], str)
            assert row["roc_auc_reason"]


def test_build_pipeline_placeholder_still_default() -> None:
    pipeline = ft.build_pipeline()
    assert pipeline.named_steps["estimator"] == "placeholder"


def test_build_pipeline_with_estimator_is_fittable(synthetic_inputs) -> None:
    columns = FEATURE_CONDITIONS["no_rfv"]
    pipeline = ft.build_pipeline(
        estimator=make_estimator("decision_tree_gini"), columns=columns
    )
    X = select_features(synthetic_inputs, "no_rfv")
    y = synthetic_inputs.y
    pipeline.fit(X.iloc[:80], y.iloc[:80])
    preds = pipeline.predict(X.iloc[80:])
    assert len(preds) == len(X) - 80
