"""Unit tests for aligned modelling inputs and feature conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.data import (
    FEATURE_CONDITIONS,
    LABELS,
    RFV_COLUMNS,
    feature_columns_for,
    prepare_modelling_inputs,
    select_features,
)
from src.preprocessing import features as ft
from src.preprocessing import sentinels as sn


def test_prepare_modelling_inputs_aligns_index(synthetic_inputs) -> None:
    assert synthetic_inputs.X.index.equals(synthetic_inputs.y.index)
    assert synthetic_inputs.X.index.equals(synthetic_inputs.groups.index)
    assert synthetic_inputs.n_rows == len(synthetic_inputs.X)
    assert synthetic_inputs.n_hospitals == len(set(synthetic_inputs.groups))


def test_hospcode_excluded_from_features(synthetic_inputs) -> None:
    assert "HOSPCODE" not in synthetic_inputs.X.columns
    assert not set(synthetic_inputs.X.columns) & set(ft.HOSPITAL_IDENTIFIERS)
    assert synthetic_inputs.groups.name == "HOSPCODE"


def test_length_mismatch_fails(synthetic_inputs) -> None:
    from src.evaluation.data import _assert_aligned

    short_y = synthetic_inputs.y.iloc[:-1]
    with pytest.raises(ValueError, match="Length mismatch"):
        _assert_aligned(synthetic_inputs.X, short_y, synthetic_inputs.groups)


def test_missing_group_column_fails(synthetic_population) -> None:
    frame = synthetic_population.drop(columns=["HOSPCODE"])
    with pytest.raises(KeyError, match="HOSPCODE"):
        prepare_modelling_inputs(frame)


def test_missing_group_value_fails(synthetic_inputs) -> None:
    from src.evaluation.data import ModellingInputs

    groups = synthetic_inputs.groups.astype(float)
    groups.iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing hospital-group value"):
        ModellingInputs(
            X=synthetic_inputs.X, y=synthetic_inputs.y, groups=groups
        )


def test_bad_target_label_fails(synthetic_population) -> None:
    frame = synthetic_population.copy()
    frame.loc[frame.index[0], "IMMEDR"] = 9
    with pytest.raises(ValueError, match="outside"):
        prepare_modelling_inputs(frame)


def test_duplicate_index_fails(synthetic_population) -> None:
    frame = synthetic_population.copy()
    frame.index = pd.Index([frame.index[0]] + list(frame.index[1:]))
    # Force an actual duplicate
    duplicated = pd.concat([frame, frame.iloc[[0]]], axis=0)
    with pytest.raises(ValueError, match="duplicates"):
        prepare_modelling_inputs(duplicated)


def test_full_vs_no_rfv_membership() -> None:
    full = feature_columns_for("full")
    no_rfv = feature_columns_for("no_rfv")
    assert set(full) == set(ft.FEATURES)
    assert set(full) - set(no_rfv) == set(RFV_COLUMNS)
    assert set(no_rfv).isdisjoint(RFV_COLUMNS)
    assert len(no_rfv) == len(full) - 5
    # Order preserved from the register
    assert no_rfv == tuple(c for c in full if c not in set(RFV_COLUMNS))


def test_select_features_no_rfv(synthetic_inputs) -> None:
    frame = select_features(synthetic_inputs, "no_rfv")
    assert list(frame.columns) == list(FEATURE_CONDITIONS["no_rfv"])
    assert not set(frame.columns) & set(RFV_COLUMNS)
    # Does not mutate source matrix
    assert set(synthetic_inputs.X.columns) == set(ft.FEATURES)


def test_unknown_condition_fails() -> None:
    with pytest.raises(KeyError, match="Unknown feature condition"):
        feature_columns_for("dropped_vitals")


def test_rfv_columns_are_canonical() -> None:
    assert RFV_COLUMNS == sn.RFV_FEATURES
    assert all(name.startswith("rfv") for name in RFV_COLUMNS)


def test_leakage_check_accepts_ablated_columns() -> None:
    ft.leakage_checks(FEATURE_CONDITIONS["no_rfv"])


def test_labels_fixed_order() -> None:
    assert LABELS == (1, 2, 3, 4, 5)


def test_groups_not_in_register(synthetic_inputs) -> None:
    # Guard against a future register change that accidentally includes HOSPCODE
    assert "HOSPCODE" not in set(ft.FEATURES)
    assert synthetic_inputs.groups.notna().all()
