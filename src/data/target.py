"""Target definition and the modelling population.

``IMMEDR`` is the NHAMCS ED Patient Record Form item "Immediacy with which
patient should be seen (unimputed)", a five-level ordinal scale.

Two codes in the released file are **not** urgency levels and are excluded on
definitional grounds, not convenience:

* ``0``  - "No triage for this visit but ESA does conduct triage". A structural
  documentation state: the facility triages, but no level was recorded for this
  visit. It is missing data *about* the target.
* ``7``  - "Visit occurred in ESA that does not conduct nursing triage". A
  different population entirely: the item does not apply, so these visits are
  off-target by construction.

Codes ``-9`` (Blank) and ``-8`` (Unknown) are non-response.

No target value is ever altered, recoded or imputed.
"""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "IMMEDR"

#: The five genuine urgency levels, in ordinal order.
URGENCY_CLASSES: tuple[int, ...] = (1, 2, 3, 4, 5)

URGENCY_LABELS: dict[int, str] = {
    1: "Immediate",
    2: "Emergent",
    3: "Urgent",
    4: "Semi-urgent",
    5: "Nonurgent",
}

#: Codes present in the released file that are not urgency levels, with the
#: documented reason each is excluded.
NON_URGENCY_CODES: dict[int, str] = {
    -9: "Blank (non-response)",
    -8: "Unknown (non-response)",
    0: "No triage recorded for this visit, but the ESA does conduct triage "
       "(structural documentation state, not an urgency level)",
    7: "Visit occurred in an ESA that does not conduct nursing triage "
       "(item not applicable - a different population)",
}


def urgency_label(code: int) -> str:
    """Human label for an urgency code. Raises rather than guessing."""
    try:
        return URGENCY_LABELS[int(code)]
    except KeyError as error:
        raise KeyError(
            f"{code!r} is not a valid urgency code; expected one of {URGENCY_CLASSES}."
        ) from error


def is_modelling_population(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting rows whose ``IMMEDR`` is a genuine urgency level."""
    return frame[TARGET_COLUMN].isin(URGENCY_CLASSES)


def select_modelling_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the modelling population, with the target column kept intact.

    Target values are not modified. Only the row set changes.
    """
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"{TARGET_COLUMN!r} not found in the loaded dataset.")
    return frame.loc[is_modelling_population(frame)].copy()


def target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Class counts and percentages over the modelling population.

    ``pct`` is always taken over the rows carrying a genuine urgency level, never
    over ``len(frame)``. That makes the function correct whether it is handed the
    full 16,025-row file or the already-filtered population, instead of silently
    reporting percentages of the wrong denominator. ``pct_of_rows_supplied`` is
    provided alongside so the distinction stays visible rather than implicit.
    """
    counts = frame[TARGET_COLUMN].value_counts()
    total_valid = int(sum(int(counts.get(code, 0)) for code in URGENCY_CLASSES))
    rows_supplied = len(frame)

    rows = [
        {
            "code": int(code),
            "label": urgency_label(code),
            "n": int(counts.get(code, 0)),
            "pct": round(100.0 * int(counts.get(code, 0)) / total_valid, 2)
            if total_valid
            else 0.0,
            "pct_of_rows_supplied": round(
                100.0 * int(counts.get(code, 0)) / rows_supplied, 2
            )
            if rows_supplied
            else 0.0,
        }
        for code in URGENCY_CLASSES
    ]
    return pd.DataFrame(rows)


def exclusion_summary(full_frame: pd.DataFrame) -> pd.DataFrame:
    """Account for every excluded row, by code, with the documented reason."""
    excluded = full_frame.loc[~is_modelling_population(full_frame), TARGET_COLUMN]
    total = len(full_frame)
    counts = excluded.value_counts()
    rows = [
        {
            "code": int(code),
            "reason": NON_URGENCY_CODES[int(code)],
            "n": int(counts.get(code, 0)),
            "pct_of_file": round(100.0 * counts.get(code, 0) / total, 2) if total else 0.0,
        }
        for code in sorted(counts.index)
    ]
    return pd.DataFrame(rows)


def majority_class_baseline(frame: pd.DataFrame) -> float:
    """Share of the most frequent urgency level - the reference any metric must beat.

    Restricted to the five genuine urgency levels, so passing the full file gives
    the same answer as passing the modelling population. On the released data this
    is 0.5232, the share of ``IMMEDR = 3`` (Urgent).
    """
    counts = frame[TARGET_COLUMN].value_counts()
    valid = [int(counts.get(code, 0)) for code in URGENCY_CLASSES]
    total = sum(valid)
    if total == 0:
        raise ValueError(f"No rows carry a valid {TARGET_COLUMN} urgency level.")
    return max(valid) / total
