"""Fixed-label classification metrics and ROC-AUC handling.

Conventions
-----------
- Class order is always :data:`src.evaluation.data.LABELS` = ``(1, 2, 3, 4, 5)``.
- ``zero_division=0`` for precision/recall/F1 so absent predictions are visible
  as zeros rather than silent NaNs.
- Fold standard deviation uses the **sample** convention (``ddof=1``). With a
  single valid fold the standard deviation is ``NaN``.
- Dummy ROC-AUC is ``N/A by protocol`` and stored as ``NaN`` with an explicit
  reason string — never as numeric zero.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.evaluation.data import LABELS

ZERO_DIVISION = 0
#: Sentinel reason recorded when ROC-AUC is not applicable by protocol.
ROC_AUC_PROTOCOL_NA = "N/A by protocol"


def align_proba_columns(
    estimator_classes: np.ndarray | list[int] | list,
    proba: np.ndarray,
    labels: tuple[int, ...] = LABELS,
) -> np.ndarray:
    """Map ``predict_proba`` columns onto the fixed ``labels`` order.

    Missing estimator classes receive an all-zero column. Callers must treat
    ROC-AUC as unreliable when a required class was absent from training.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {proba.shape}.")
    if proba.shape[1] != len(estimator_classes):
        raise ValueError(
            f"proba has {proba.shape[1]} columns but estimator has "
            f"{len(estimator_classes)} classes."
        )
    class_to_pos = {
        int(label): position for position, label in enumerate(estimator_classes)
    }
    aligned = np.zeros((proba.shape[0], len(labels)), dtype=float)
    for column, label in enumerate(labels):
        position = class_to_pos.get(int(label))
        if position is not None:
            aligned[:, column] = proba[:, position]
    return aligned


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(LABELS),
        zero_division=ZERO_DIVISION,
    )
    per_class = {
        int(label): {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
        }
        for position, label in enumerate(LABELS)
    }
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=ZERO_DIVISION,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": per_class,
    }


def fold_roc_auc(
    y_true: np.ndarray,
    proba_aligned: np.ndarray | None,
    *,
    model_id: str,
    labels: tuple[int, ...] = LABELS,
) -> tuple[float, str]:
    """Return ``(auc, reason)`` for one fold.

    ``auc`` is ``NaN`` when unavailable and ``reason`` then states why;
    ``reason`` is empty when AUC was computed successfully.

    Class-presence requirements for the fixed five-class, one-vs-rest macro
    ROC-AUC, checked in this order:

    1. Fewer than two distinct classes in the test fold (the degenerate
       single-class case) - ROC-AUC is undefined outright.
    2. A test-present class with a constant probability column (the class was
       never seen in training) - its one-vs-rest sub-score is undefined.
    3. Any class of ``labels`` absent from the test fold - a class with no
       positive examples has no defined one-vs-rest sub-score, and the macro
       average is over all five classes, so the fold's AUC is undefined.

    Dummy models are N/A by protocol. In every unavailable case the
    classification metrics are still computed and unaffected.
    """
    if model_id.startswith("dummy"):
        return float("nan"), ROC_AUC_PROTOCOL_NA

    if proba_aligned is None:
        return float("nan"), "predict_proba unavailable"

    y_true = np.asarray(y_true)
    proba_aligned = np.asarray(proba_aligned, dtype=float)
    if proba_aligned.shape != (y_true.shape[0], len(labels)):
        return (
            float("nan"),
            f"probability shape {proba_aligned.shape} does not match "
            f"(n={y_true.shape[0]}, n_labels={len(labels)})",
        )

    present = [label for label in labels if int((y_true == label).sum()) > 0]
    if len(present) < 2:
        return (
            float("nan"),
            f"undefined: test fold contains {len(present)} class(es) "
            f"{present}; ROC-AUC needs at least two distinct classes",
        )

    # A test-present class whose probability column is constant - typically
    # because the class never appeared in the training fold - makes that
    # one-vs-rest sub-score undefined. Classes absent from the test fold are
    # rejected by the required-class check below.
    constant_columns = [
        int(labels[column])
        for column in range(proba_aligned.shape[1])
        if np.unique(proba_aligned[:, column]).size <= 1
        and int(labels[column]) in present
    ]
    if constant_columns:
        return (
            float("nan"),
            "undefined: constant probability column(s) for test-present "
            f"class(es) {constant_columns} (absent from training)",
        )

    # The macro average runs over every class of the fixed label set, and a
    # class with no positive examples in the test fold has an undefined
    # one-vs-rest sub-score. Requiring all five classes is a stronger
    # condition than the "at least two distinct classes" rule above.
    missing = [label for label in labels if label not in present]
    if missing:
        return (
            float("nan"),
            f"undefined: test fold is missing required class(es) {missing}; "
            f"fixed five-class macro OvR ROC-AUC needs every class in "
            f"{list(labels)} present in the test fold",
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            auc = roc_auc_score(
                y_true,
                proba_aligned,
                labels=list(labels),
                multi_class="ovr",
                average="macro",
            )
        except ValueError as error:
            return float("nan"), f"undefined: {error}"
        except Warning as warning:
            # Any further scikit-learn warning (promoted to an error here)
            # likewise means the reported AUC would not be trustworthy.
            return float("nan"), f"undefined: {warning}"
    if auc is None or not np.isfinite(auc):
        return float("nan"), "undefined: non-finite ROC-AUC"
    return float(auc), ""


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_id: str,
    proba_aligned: np.ndarray | None = None,
) -> dict[str, Any]:
    """Classification metrics plus optional ROC-AUC for one fold."""
    metrics = _classification_metrics(y_true, y_pred)
    auc, auc_reason = fold_roc_auc(
        y_true, proba_aligned, model_id=model_id
    )
    metrics["roc_auc"] = auc
    metrics["roc_auc_reason"] = auc_reason
    metrics["roc_auc_available"] = bool(np.isfinite(auc))
    return metrics


def per_class_frame(
    metrics: dict[str, Any],
    *,
    model: str,
    condition: str,
    fold: int | str,
) -> pd.DataFrame:
    """Flatten per-class precision/recall/F1 into a long DataFrame."""
    rows = []
    for label in LABELS:
        block = metrics["per_class"][int(label)]
        rows.append(
            {
                "model": model,
                "condition": condition,
                "fold": fold,
                "class": int(label),
                "precision": block["precision"],
                "recall": block["recall"],
                "f1": block["f1"],
            }
        )
    return pd.DataFrame(rows)


def pooled_metrics_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model: str,
    condition: str,
) -> dict[str, Any]:
    """Pooled out-of-fold classification metrics (not a fold average)."""
    metrics = _classification_metrics(y_true, y_pred)
    metrics["model"] = model
    metrics["condition"] = condition
    metrics["n_oof"] = int(len(y_true))
    return metrics


def mean_std_summary(values: list[float]) -> dict[str, float | int]:
    """Sample mean/std (``ddof=1``) and count of finite entries."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    n_valid = int(finite.size)
    mean = float(finite.mean()) if n_valid else float("nan")
    std = float(finite.std(ddof=1)) if n_valid > 1 else float("nan")
    return {"mean": mean, "std": std, "n_valid": n_valid}


def confusion_frame(
    y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Confusion matrix with explicit rows=true, columns=predicted, labels 1–5."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    return pd.DataFrame(
        matrix,
        index=pd.Index(LABELS, name="true"),
        columns=pd.Index(LABELS, name="predicted"),
    )
