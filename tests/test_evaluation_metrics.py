"""Unit tests for metrics, probability alignment, and ROC-AUC policy."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.data import LABELS
from src.evaluation.metrics import (
    ROC_AUC_PROTOCOL_NA,
    align_proba_columns,
    compute_fold_metrics,
    confusion_frame,
    fold_roc_auc,
    mean_std_summary,
)


def test_align_proba_reorders_columns() -> None:
    # Estimator saw classes [3, 1, 5] in that order.
    estimator_classes = np.array([3, 1, 5])
    proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
        ]
    )
    aligned = align_proba_columns(estimator_classes, proba, LABELS)
    assert aligned.shape == (2, 5)
    np.testing.assert_allclose(aligned[0, 0], 0.2)  # class 1
    np.testing.assert_allclose(aligned[0, 1], 0.0)  # class 2 missing
    np.testing.assert_allclose(aligned[0, 2], 0.7)  # class 3
    np.testing.assert_allclose(aligned[0, 4], 0.1)  # class 5


def test_align_proba_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="columns"):
        align_proba_columns([1, 2], np.zeros((4, 3)))


def test_dummy_roc_auc_is_protocol_na() -> None:
    y_true = np.array([1, 2, 3, 4, 5, 1, 2, 3])
    y_pred = np.array([3, 3, 3, 3, 3, 3, 3, 3])
    metrics = compute_fold_metrics(
        y_true, y_pred, model_id="dummy_most_frequent", proba_aligned=None
    )
    assert np.isnan(metrics["roc_auc"])
    assert metrics["roc_auc_reason"] == ROC_AUC_PROTOCOL_NA
    assert metrics["roc_auc_available"] is False


def test_roc_auc_undefined_single_class_test_fold() -> None:
    y_true = np.array([3, 3, 3, 3])
    proba = np.full((4, 5), 0.2)
    # Make columns non-constant so the single-class rule fires first.
    proba[:, 0] = np.linspace(0.1, 0.4, 4)
    auc, reason = fold_roc_auc(y_true, proba, model_id="gaussian_nb")
    assert np.isnan(auc)
    assert "at least two" in reason
    # Distinguished from the missing-fixed-class case below.
    assert "missing required class" not in reason


def test_roc_auc_undefined_missing_required_class() -> None:
    """Every class of the fixed label set must appear in the test fold.

    The macro one-vs-rest average runs over all five classes, so a test fold
    lacking class 5 has no defined AUC even though four classes are present
    and every probability column behaves.
    """
    y_true = np.array([1, 1, 2, 2, 3, 3, 4, 4])  # class 5 absent from test
    proba = np.full((8, 5), 0.1)
    for i, label in enumerate(y_true):
        proba[i, label - 1] = 0.6  # rows sum to 0.6 + 4 * 0.1 = 1.0
    auc, reason = fold_roc_auc(y_true, proba, model_id="gaussian_nb")
    assert np.isnan(auc)
    assert "missing required class" in reason
    assert "[5]" in reason
    # Distinguished from the degenerate fewer-than-two-classes case.
    assert "at least two" not in reason

    # Classification metrics stay available next to the undefined AUC.
    metrics = compute_fold_metrics(
        y_true, y_true.copy(), model_id="gaussian_nb", proba_aligned=proba
    )
    assert np.isnan(metrics["roc_auc"])
    assert "missing required class" in metrics["roc_auc_reason"]
    assert np.isfinite(metrics["macro_f1"])


def test_roc_auc_undefined_constant_probability_column() -> None:
    # Classes 1 and 5 present in test; class 5 probability constant (unseen in train).
    y_true = np.array([1, 1, 5, 5])
    proba = np.zeros((4, 5))
    proba[:, 0] = [0.9, 0.8, 0.1, 0.2]  # class 1 varies
    proba[:, 4] = [0.0, 0.0, 0.0, 0.0]  # class 5 constant
    auc, reason = fold_roc_auc(y_true, proba, model_id="gaussian_nb")
    assert np.isnan(auc)
    assert "constant probability" in reason


def test_roc_auc_computes_when_valid() -> None:
    y_true = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    proba = np.zeros((10, 5))
    for i, label in enumerate(y_true):
        proba[i, label - 1] = 0.8
        for other in range(5):
            if other != label - 1:
                proba[i, other] = 0.2 / 4
    auc, reason = fold_roc_auc(y_true, proba, model_id="gaussian_nb")
    assert reason == ""
    assert np.isfinite(auc)
    assert auc > 0.9


def test_fixed_label_metrics_include_absent_class() -> None:
    # Never predicts class 5; class 5 absent from y_true as well.
    y_true = np.array([1, 1, 2, 2, 3, 3])
    y_pred = np.array([1, 2, 2, 2, 3, 1])
    metrics = compute_fold_metrics(y_true, y_pred, model_id="decision_tree_gini")
    assert set(metrics["per_class"]) == set(LABELS)
    assert metrics["per_class"][5]["precision"] == 0.0
    assert metrics["per_class"][5]["recall"] == 0.0
    assert metrics["per_class"][5]["f1"] == 0.0
    # Macro-F1 still averages all five labels
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_mean_std_sample_convention() -> None:
    stats = mean_std_summary([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))
    assert stats["n_valid"] == 3


def test_mean_std_single_valid_fold_is_nan_std() -> None:
    stats = mean_std_summary([0.5, float("nan")])
    assert stats["mean"] == pytest.approx(0.5)
    assert np.isnan(stats["std"])
    assert stats["n_valid"] == 1


def test_mean_std_all_invalid() -> None:
    stats = mean_std_summary([float("nan"), float("nan")])
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["std"])
    assert stats["n_valid"] == 0


def test_confusion_frame_shape_and_labels() -> None:
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 2, 4, 5])
    frame = confusion_frame(y_true, y_pred)
    assert frame.shape == (5, 5)
    assert list(frame.index) == list(LABELS)
    assert list(frame.columns) == list(LABELS)
    assert int(frame.loc[2, 2]) == 1
    assert int(frame.loc[3, 2]) == 1
