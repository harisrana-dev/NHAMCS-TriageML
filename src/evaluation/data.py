"""Aligned modelling inputs and the two approved feature conditions.

The evaluation layer never reimplements Part 1 cleaning. It consumes the
modelling population frame produced by
:func:`src.data.target.select_modelling_population` (which retains ``HOSPCODE``
and ``IMMEDR``), projects it with
:func:`src.preprocessing.sentinels.build_modelling_frame`, and asserts index
alignment before any split is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.data import target as tgt
from src.preprocessing import features as ft
from src.preprocessing import sentinels as sn

#: Fixed five-class order used for every metric, probability column, and report.
LABELS: tuple[int, ...] = tgt.URGENCY_CLASSES

#: Canonical RFV complaint columns in the engineered modelling frame.
RFV_COLUMNS: tuple[str, ...] = sn.RFV_FEATURES

#: Approved feature conditions. Keys are stable experiment identifiers.
FEATURE_CONDITIONS: Mapping[str, tuple[str, ...]] = {
    "full": tuple(ft.FEATURES),
    "no_rfv": tuple(
        column for column in ft.FEATURES if column not in set(RFV_COLUMNS)
    ),
}

GROUP_COLUMN = "HOSPCODE"


@dataclass(frozen=True)
class ModellingInputs:
    """Index-aligned feature matrix, target, and hospital groups."""

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series

    def __post_init__(self) -> None:
        _assert_aligned(self.X, self.y, self.groups)
        if self.groups.isna().any():
            n_missing = int(self.groups.isna().sum())
            raise ValueError(
                f"groups contains {n_missing} missing hospital-group value(s); "
                "every row must carry a group code so grouped splitting can "
                "assign it to exactly one fold."
            )
        if any(column in self.X.columns for column in ft.HOSPITAL_IDENTIFIERS):
            leaked = sorted(set(self.X.columns) & set(ft.HOSPITAL_IDENTIFIERS))
            raise ValueError(
                f"Hospital/survey identifiers must not appear in X: {leaked}"
            )
        unexpected = sorted(set(self.X.columns) - set(ft.FEATURES))
        if unexpected:
            raise ValueError(
                f"X contains columns outside the approved register: {unexpected}"
            )
        missing = sorted(set(ft.FEATURES) - set(self.X.columns))
        if missing:
            raise ValueError(f"X is missing approved register columns: {missing}")
        labels = set(pd.unique(self.y))
        if not labels.issubset(set(LABELS)):
            raise ValueError(
                f"y contains labels outside {LABELS}: {sorted(labels - set(LABELS))}"
            )

    @property
    def n_rows(self) -> int:
        return len(self.X)

    @property
    def n_hospitals(self) -> int:
        return int(pd.unique(self.groups).size)


def _assert_aligned(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> None:
    """Require identical lengths and identical index objects/values."""
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(
            f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}"
        )
    if len(X) == 0:
        raise ValueError("Modelling inputs are empty.")
    if not X.index.equals(y.index):
        raise ValueError("X.index and y.index are not aligned.")
    if not X.index.equals(groups.index):
        raise ValueError("X.index and groups.index are not aligned.")
    if X.index.has_duplicates:
        raise ValueError("Row index contains duplicates; alignment is ambiguous.")


def prepare_modelling_inputs(
    population: pd.DataFrame,
    group_column: str = GROUP_COLUMN,
) -> ModellingInputs:
    """Build aligned ``X``, ``y``, and hospital ``groups`` from a population frame.

    Parameters
    ----------
    population:
        Output of :func:`src.data.target.select_modelling_population` — the
        raw NHAMCS modelling rows with ``IMMEDR`` intact and ``HOSPCODE``
        still present for grouping.
    group_column:
        Hospital identifier used only for splitting. Defaults to
        ``HOSPCODE``.
    """
    if tgt.TARGET_COLUMN not in population.columns:
        raise KeyError(f"Target column {tgt.TARGET_COLUMN!r} not found.")
    if group_column not in population.columns:
        raise KeyError(f"Group column {group_column!r} not found.")
    if group_column in set(ft.FEATURES):
        raise ValueError(
            f"Group column {group_column!r} is an approved predictive feature; "
            "groups must not be predictive features."
        )

    X = sn.build_modelling_frame(population)
    y = population[tgt.TARGET_COLUMN].copy()
    groups = population[group_column].copy()
    y.name = tgt.TARGET_COLUMN
    groups.name = group_column
    _assert_aligned(X, y, groups)
    return ModellingInputs(X=X, y=y, groups=groups)


def feature_columns_for(condition: str) -> tuple[str, ...]:
    """Return the approved column tuple for a named feature condition."""
    try:
        return FEATURE_CONDITIONS[condition]
    except KeyError as error:
        known = ", ".join(sorted(FEATURE_CONDITIONS))
        raise KeyError(
            f"Unknown feature condition {condition!r}. Expected one of: {known}."
        ) from error


def select_features(inputs: ModellingInputs, condition: str) -> pd.DataFrame:
    """Return the feature matrix for ``condition`` without mutating ``inputs``."""
    columns = feature_columns_for(condition)
    frame = inputs.X.loc[:, list(columns)].copy()
    frame.columns = list(columns)
    return frame
