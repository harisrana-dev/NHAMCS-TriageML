"""Shared pytest configuration for NHAMCS-TriageML.

Ensures the repository root is importable as the ``src`` package parent,
registers the ``integration`` marker, and provides synthetic fixtures that
mirror the Part 1 modelling-frame schema without patient records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that require the pinned NHAMCS dataset on disk",
    )


RNG_SEED = 7
N_ROWS = 120
N_HOSPITALS = 12


def synthetic_modelling_frame(
    n_rows: int = N_ROWS, n_hospitals: int = N_HOSPITALS
) -> pd.DataFrame:
    """Build a synthetic **raw** population frame (Part 1 source columns).

    Column names follow the 18 raw source variables in
    ``src.preprocessing.features.RAW_SOURCE_VARIABLES`` plus ``HOSPCODE`` and
    ``IMMEDR``, so :func:`prepare_modelling_inputs` can run the real
    ``build_modelling_frame`` path.
    """
    rng = np.random.default_rng(RNG_SEED)
    indices = np.arange(1000, 1000 + n_rows)
    hospitals = rng.integers(1, n_hospitals + 1, size=n_rows)
    labels = np.resize(np.array([1, 2, 3, 4, 5], dtype=int), n_rows)
    rng.shuffle(labels)

    frame = pd.DataFrame(index=indices)
    # ARRTIME as HHMM strings (valid military times).
    hours = rng.integers(0, 24, size=n_rows)
    minutes = rng.integers(0, 60, size=n_rows)
    frame["ARRTIME"] = [f"{h:02d}{m:02d}" for h, m in zip(hours, minutes)]
    frame["TEMPF"] = rng.integers(912, 1051, size=n_rows).astype(float)  # x10 scale
    frame["PULSE"] = rng.integers(50, 130, size=n_rows).astype(float)
    frame["RESPR"] = rng.integers(10, 30, size=n_rows).astype(float)
    frame["BPSYS"] = rng.integers(90, 180, size=n_rows).astype(float)
    frame["BPDIAS"] = rng.integers(50, 110, size=n_rows).astype(float)
    frame["POPCT"] = rng.integers(92, 101, size=n_rows).astype(float)
    frame["VMONTH"] = rng.integers(1, 13, size=n_rows).astype(float)
    frame["VDAYR"] = rng.integers(1, 8, size=n_rows).astype(float)
    frame["RESIDNCE"] = rng.choice([1, 2, 3, -9, -8], size=n_rows).astype(float)
    frame["ARREMS"] = rng.choice([1, 2, -9], size=n_rows).astype(float)
    frame["AMBTRANSFER"] = rng.choice([1, 2, -7, -9], size=n_rows).astype(float)
    frame["SEEN72"] = rng.choice([1, 2, -9, -8], size=n_rows).astype(float)
    for position, column in enumerate(
        ["RFV13D", "RFV23D", "RFV33D", "RFV43D", "RFV53D"]
    ):
        codes = list(range(100, 100 + 8 + position)) + [0]
        frame[column] = rng.choice(codes, size=n_rows).astype(float)

    frame["HOSPCODE"] = hospitals.astype(np.int16)
    frame["IMMEDR"] = labels.astype(np.int16)
    return frame


@pytest.fixture()
def synthetic_population() -> pd.DataFrame:
    return synthetic_modelling_frame()


@pytest.fixture()
def synthetic_inputs(synthetic_population):
    from src.evaluation.data import prepare_modelling_inputs

    return prepare_modelling_inputs(synthetic_population)


@pytest.fixture()
def tiny_population() -> pd.DataFrame:
    """Very small population for exact arithmetic checks."""
    return synthetic_modelling_frame(n_rows=30, n_hospitals=5)
