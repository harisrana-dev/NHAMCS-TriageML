"""Hospital-grouped cross-validation splits.

Splits are generated once per evaluation run and reused for every model and
feature condition so that reported metrics are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.evaluation.data import LABELS


@dataclass(frozen=True)
class GroupedFold:
    """One train/test split with positional indices and hospital audit info."""

    fold_id: int
    train_positions: np.ndarray
    test_positions: np.ndarray
    train_hospitals: frozenset
    test_hospitals: frozenset
    test_class_counts: dict[int, int]


def make_grouped_folds(
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int | None = None,
    shuffle: bool = True,
) -> list[GroupedFold]:
    """Create validated ``StratifiedGroupKFold`` splits on hospital groups.

    Uses :data:`src.config.RANDOM_STATE` when ``random_state`` is ``None`` so
    Part 1 and Part 2 share one seed unless a caller overrides it explicitly.
    """
    if random_state is None:
        random_state = config.RANDOM_STATE
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

    y_values = np.asarray(y)
    group_values = np.asarray(groups)
    if len(y_values) != len(group_values):
        raise ValueError(
            f"y and groups length mismatch: {len(y_values)} vs {len(group_values)}"
        )
    if len(y_values) == 0:
        raise ValueError("Cannot split empty inputs.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    folds: list[GroupedFold] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(y_values)), y_values, group_values)
    ):
        train_hospitals = frozenset(group_values[train_idx].tolist())
        test_hospitals = frozenset(group_values[test_idx].tolist())
        test_counts = {
            int(label): int((y_values[test_idx] == label).sum()) for label in LABELS
        }
        folds.append(
            GroupedFold(
                fold_id=fold_id,
                train_positions=np.asarray(train_idx, dtype=np.int64),
                test_positions=np.asarray(test_idx, dtype=np.int64),
                train_hospitals=train_hospitals,
                test_hospitals=test_hospitals,
                test_class_counts=test_counts,
            )
        )
    validate_folds(folds, n_splits=n_splits, n_rows=len(y_values))
    return folds


def validate_folds(
    folds: list[GroupedFold],
    *,
    n_splits: int,
    n_rows: int,
) -> None:
    """Assert group isolation, positional coverage, and basic fold hygiene."""
    if len(folds) != n_splits:
        raise ValueError(f"Expected {n_splits} folds, got {len(folds)}.")

    seen_test_positions: list[np.ndarray] = []
    for fold in folds:
        overlap = fold.train_hospitals & fold.test_hospitals
        if overlap:
            preview = sorted(list(overlap))[:10]
            raise ValueError(
                f"Fold {fold.fold_id} has {len(overlap)} hospital(s) in both "
                f"train and test (e.g. {preview})."
            )
        if len(fold.train_positions) == 0 or len(fold.test_positions) == 0:
            raise ValueError(f"Fold {fold.fold_id} has an empty train or test set.")
        train_set = set(fold.train_positions.tolist())
        test_set = set(fold.test_positions.tolist())
        if train_set & test_set:
            raise ValueError(f"Fold {fold.fold_id} has overlapping row positions.")
        if not test_set.issubset(range(n_rows)) or not train_set.issubset(
            range(n_rows)
        ):
            raise ValueError(f"Fold {fold.fold_id} positions out of range.")
        seen_test_positions.append(fold.test_positions)

    concatenated = np.concatenate(seen_test_positions)
    if concatenated.size != n_rows:
        raise ValueError(
            f"OOF coverage incomplete: test positions cover {concatenated.size} "
            f"rows, expected {n_rows}."
        )
    if np.unique(concatenated).size != n_rows:
        unique, counts = np.unique(concatenated, return_counts=True)
        duplicates = unique[counts > 1]
        raise ValueError(
            f"Test positions are not exact-once: duplicate position(s) "
            f"{duplicates.tolist()}."
        )


def fold_diagnostics(folds: list[GroupedFold]) -> pd.DataFrame:
    """Per-fold class presence and hospital counts for audit trails."""
    rows = []
    for fold in folds:
        present = [label for label, count in fold.test_class_counts.items() if count > 0]
        missing = [label for label, count in fold.test_class_counts.items() if count == 0]
        rows.append(
            {
                "fold": fold.fold_id,
                "n_train": int(fold.train_positions.size),
                "n_test": int(fold.test_positions.size),
                "n_train_hospitals": len(fold.train_hospitals),
                "n_test_hospitals": len(fold.test_hospitals),
                "hospital_overlap": len(fold.train_hospitals & fold.test_hospitals),
                "test_classes_present": tuple(present),
                "test_classes_absent": tuple(missing),
                "test_class_counts": dict(fold.test_class_counts),
            }
        )
    return pd.DataFrame(rows)
