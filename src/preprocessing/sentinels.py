"""Codebook-derived sentinel handling and range validation for NHAMCS 2022 ED.

Design rule
-----------
NHAMCS uses a **variable-specific** sentinel scheme. There is no single "negative
means missing" rule, and applying one is an error. Three distinct situations
occur, and each gets its own treatment:

=========================================  ==========================  ====================
Situation                                   Treatment                  Example
=========================================  ==========================  ====================
A measurement was **not recorded**          ``NaN`` + missing indicator ``TEMPF = -9``
A **response state was recorded**           explicit labelled category  ``RESIDNCE = -8`` -> ``unknown``
The item is **structurally not applicable** explicit labelled category  ``AMBTRANSFER = -7``
=========================================  ==========================  ====================

Only the first case is imputed. Imputing across the second or third would
fabricate a response or an applicability that was never given.

**Every function in this module is row-local and stateless.** Nothing here
computes a statistic, a frequency or a threshold across rows, so cleaning can
never leak information across a train/validation boundary.

Sources of record
-----------------
* Value-label semantics read directly from the official Stata file
  (``PULSEF``, ``BPDIASF``, ``ARREMSF``, ``IMMEDRF``, ...).
* ``ed2022.do`` / ``eddict2022.dct``: ``str ARRTIME 4-7``; ``long RFV1 73-77``
  versus ``long RFV13D 73-76``, confirming the 4-digit truncation.
* Documented ranges transcribed in the forensic audit
  (``audit/range_forensics.csv``) from ``doc22-ed-508.pdf``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Sentinel vocabulary ------------------------------------------------------

BLANK = -9  # "Blank"
UNKNOWN = -8  # "Unknown"
NOT_APPLICABLE = -7  # "Not applicable"

SENTINEL_LABELS: dict[int, str] = {
    BLANK: "blank",
    UNKNOWN: "unknown",
    NOT_APPLICABLE: "not_applicable",
}

#: Reason-for-visit slot absent -> structural code 0 in the 4-digit variables.
#: Verified: RFV<D>3D == 0 holds exactly when the 5-digit RFV<n> is blank
#: (10,207 / 10,207 rows, no off-diagonal cells). Encoding absence as 0 rather
#: than -9 is a formatting choice, not a measurement.
RFV_ABSENT_CODE = 0
RFV_ABSENT_LABEL = "none_recorded"

# --- Method / applicability codes that are not measurements ------------------

#: ``PULSEF`` value-label set defines 998 = "DOPP or DOPPLER".
PULSE_METHOD_CODE = 998
#: ``BPDIASF`` value-label set defines 998 = "P, Palp, DOP or DOPPLER".
BPDIAS_METHOD_CODE = 998
#: ``BPSYS`` documented as 0 = unobtainable (no value-label entry; doc note).
BPSYS_UNOBTAINABLE_CODE = 0

#: TEMPF carries **no** method code. TEMPF is an implied-decimal (x10) scale, so
#: 998 means 99.8 F - a valid temperature. Treating it as a method code, as a
#: blanket "998 = Doppler" rule would, would destroy real data.
TEMPF_SCALE_DIVISOR = 10

# --- Engineered feature names -------------------------------------------------
# Renaming makes the unit and the meaning explicit in the transformed matrix.

ARRIVAL_MINUTES = "arrival_minutes"
TEMP_F = "temp_f"
PULSE_BPM = "pulse_bpm"
PULSE_METHOD = "pulse_measured_by_doppler"
RESP_RATE = "resp_rate"
BP_SYSTOLIC = "bp_systolic"
BP_DIASTOLIC = "bp_diastolic"
BP_DIASTOLIC_METHOD = "bp_diastolic_palpated_or_doppler"
SPO2_PCT = "spo2_pct"

VISIT_MONTH = "visit_month"
VISIT_DOW = "visit_day_of_week"
PATIENT_RESIDENCE = "patient_residence"
ARRIVAL_BY_AMBULANCE = "arrival_by_ambulance"
INTERFACILITY_TRANSFER = "interfacility_transfer"
SEEN_WITHIN_72H = "seen_in_this_ed_within_72h"

RFV_FEATURES: tuple[str, ...] = (
    "rfv1_broad_category",
    "rfv2_broad_category",
    "rfv3_broad_category",
    "rfv4_broad_category",
    "rfv5_broad_category",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    ARRIVAL_MINUTES,
    TEMP_F,
    PULSE_BPM,
    RESP_RATE,
    BP_SYSTOLIC,
    BP_DIASTOLIC,
    SPO2_PCT,
)

#: Binary method indicators. Numeric, but not measurements - kept separate from
#: NUMERIC_FEATURES so the scaling block does not standardise a 0/1 flag.
METHOD_FLAG_FEATURES: tuple[str, ...] = (
    PULSE_METHOD,
    BP_DIASTOLIC_METHOD,
)

LOW_CARDINALITY_CATEGORICAL: tuple[str, ...] = (
    VISIT_MONTH,
    VISIT_DOW,
    PATIENT_RESIDENCE,
    ARRIVAL_BY_AMBULANCE,
    INTERFACILITY_TRANSFER,
    SEEN_WITHIN_72H,
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    LOW_CARDINALITY_CATEGORICAL + RFV_FEATURES
)

#: Everything the ColumnTransformer is allowed to see: 7 measurements, 2 binary
#: method indicators, and 11 categorical codes. 20 columns in total, derived from
#: 18 raw source variables (PULSE and BPDIAS each contribute a value and a flag).
MODELLING_FEATURES: tuple[str, ...] = (
    NUMERIC_FEATURES + METHOD_FLAG_FEATURES + CATEGORICAL_FEATURES
)

# --- Documented ranges (transcribed from doc22-ed-508.pdf via the audit) ------
# (low, high, provenance). Used for VALIDATION ONLY - nothing is clipped.

DOCUMENTED_RANGES: dict[str, tuple[float, float, str]] = {
    ARRIVAL_MINUTES: (0.0, 1439.0, "ARRTIME military time 0000-2359 (eddict2022.dct: str ARRTIME 4-7)"),
    TEMP_F: (89.6, 105.6, "TEMPF implied-decimal scale 896-1056 (x10)"),
    PULSE_BPM: (0.0, 240.0, "PULSE reported range 0-240; 998 is a method code, not a value"),
    RESP_RATE: (0.0, 150.0, "RESPR reported range 0-150"),
    BP_SYSTOLIC: (43.0, 289.0, "BPSYS reported range 43-289; 0 = unobtainable"),
    BP_DIASTOLIC: (22.0, 190.0, "BPDIAS reported range 22-190; 998 is a method code, not a value"),
    SPO2_PCT: (0.0, 100.0, "POPCT reported range 0-100"),
}

#: Values that are inside the documented range but clinically implausible.
#: Reported, never altered.
FLAGGED_IMPLAUSIBLE: dict[str, dict[int, str]] = {
    SPO2_PCT: {0: "POPCT = 0 % is inside the documented range 0-100 but is not a "
                  "physiologically survivable saturation. Retained and reported; "
                  "not clipped, not imputed."},
}

#: Human-readable statement of every special-code decision, for the notebook.
SENTINEL_DECISIONS: tuple[dict[str, str], ...] = (
    {
        "source": "ARRTIME",
        "engineered": ARRIVAL_MINUTES,
        "observed_code": "literal '-9'",
        "treatment": "-> NaN (not recorded)",
        "why": "243 such values in the full file. Parsed as HH*60+MM so the value "
               "is comparable across the day.",
    },
    {
        "source": "ARRTIME",
        "engineered": ARRIVAL_MINUTES,
        "observed_code": "'0000'",
        "treatment": "-> 0 (retained)",
        "why": "Midnight arrival is a legitimate clock time, not a missing value.",
    },
    {
        "source": "TEMPF",
        "engineered": TEMP_F,
        "observed_code": "-9",
        "treatment": "-> NaN + indicator",
        "why": "Temperature not recorded.",
    },
    {
        "source": "TEMPF",
        "engineered": TEMP_F,
        "observed_code": "998",
        "treatment": "-> 99.8 F (retained as a value)",
        "why": "TEMPF has NO method code. It is an implied-decimal x10 scale, so "
               "998 is 99.8 F. A blanket '998 = Doppler' rule would corrupt this.",
    },
    {
        "source": "PULSE",
        "engineered": PULSE_BPM,
        "observed_code": "-9",
        "treatment": "-> NaN + indicator",
        "why": "Heart rate not recorded.",
    },
    {
        "source": "PULSE",
        "engineered": PULSE_METHOD,
        "observed_code": "998",
        "treatment": "-> NaN + binary indicator",
        "why": "Value-label set PULSEF defines 998 = 'DOPP or DOPPLER'. A "
               "measurement method, not a heart rate. 88 records in the full file.",
    },
    {
        "source": "BPSYS",
        "engineered": BP_SYSTOLIC,
        "observed_code": "-9",
        "treatment": "-> NaN + indicator",
        "why": "Systolic pressure not recorded.",
    },
    {
        "source": "BPSYS",
        "engineered": BP_SYSTOLIC,
        "observed_code": "0",
        "treatment": "-> NaN + indicator",
        "why": "Documented as unobtainable. Not a pressure of 0 mmHg.",
    },
    {
        "source": "BPDIAS",
        "engineered": BP_DIASTOLIC,
        "observed_code": "-9 / 998",
        "treatment": "-> NaN + binary indicator",
        "why": "Value-label set BPDIASF defines 998 = 'P, Palp, DOP or DOPPLER'.",
    },
    {
        "source": "AMBTRANSFER",
        "engineered": INTERFACILITY_TRANSFER,
        "observed_code": "-7",
        "treatment": "-> category 'not_applicable' (retained)",
        "why": "The item only applies to ambulance arrivals, so -7 is structural "
               "(~81% of rows), not random missingness. Imputing across it would "
               "invent a transfer answer.",
    },
    {
        "source": "RESIDNCE / ARREMS / SEEN72",
        "engineered": "categorical",
        "observed_code": "-9 / -8",
        "treatment": "-> categories 'blank' / 'unknown' (retained)",
        "why": "These are recorded response states, so imputing the mode would "
               "fabricate a response that was never given.",
    },
    {
        "source": "RFV13D..RFV53D",
        "engineered": "rfv*_broad_category",
        "observed_code": "0",
        "treatment": "-> category 'none_recorded' (retained)",
        "why": "Structural: verified to hold exactly when the corresponding 5-digit "
               "RFV<n> is blank (10,207/10,207 rows). Imputing would invent a "
               "reason for visit.",
    },
)


# --- Row-local cleaning primitives -------------------------------------------


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_arrival_time(series: pd.Series) -> pd.Series:
    """Parse ``HHMM`` military time to minutes since midnight.

    Literal ``-9`` becomes ``NaN``. ``'0000'`` becomes ``0`` and is kept: it is a
    real midnight arrival.
    """
    text = series.astype("string").str.strip()
    text = text.mask(text.isin({"-9", "-8", "-7"}), other=pd.NA)
    hours = pd.to_numeric(text.str.slice(0, 2), errors="coerce")
    minutes = pd.to_numeric(text.str.slice(2, 4), errors="coerce")
    total = hours * 60 + minutes
    # A value is only usable if it is a well-formed 24-hour clock time.
    invalid = total.isna() | (hours > 23) | (minutes > 59)
    return total.mask(invalid)


def _blank_to_nan(series: pd.Series) -> pd.Series:
    """Map survey non-response sentinels (-9, -8) to ``NaN`` for measurements."""
    numeric = _to_numeric(series)
    return numeric.mask(numeric.isin([BLANK, UNKNOWN]))


def _categorical_with_labels(
    series: pd.Series,
    structural_absent: int | None = None,
    absent_label: str | None = None,
) -> pd.Series:
    """Render a coded response item as a labelled string category.

    Three things that are *not* missing data get their own explicit category, and
    are never imputed:

    * recorded response states (``-9`` blank, ``-8`` unknown, ``-7`` not
      applicable), per :data:`SENTINEL_LABELS`;
    * a structural "this item does not exist for this visit" code, such as
      ``0`` in the 4-digit reason-for-visit variables;
    * genuine source ``NaN``, which becomes ``"missing"``.

    Order matters: structural codes must be recognised before the generic
    sentinel map, otherwise an item that is legitimately not-applicable would be
    reported as missing data and then imputed across.
    """
    numeric = _to_numeric(series)
    out = pd.Series(pd.NA, index=numeric.index, dtype="string")

    # 1. Real observed codes.
    reserved = set(SENTINEL_LABELS)
    if structural_absent is not None:
        reserved.add(structural_absent)
    is_real = ~numeric.isna() & ~numeric.isin(reserved)
    out.loc[is_real] = numeric.loc[is_real].astype("int64").astype("string")

    # 2. Structural absence for this visit.
    if structural_absent is not None:
        label = absent_label or SENTINEL_LABELS.get(structural_absent, "not_applicable")
        out.loc[numeric == structural_absent] = label

    # 3. Recorded response states.
    for code, label in SENTINEL_LABELS.items():
        if code == structural_absent:
            continue  # already handled above under the structural rule
        out.loc[numeric == code] = label

    # 4. Genuine source missingness.
    out.loc[numeric.isna()] = "missing"

    return out


def _measurement_with_method_flag(
    series: pd.Series, method_code: int | None, unobtainable_code: int | None = None
) -> tuple[pd.Series, pd.Series]:
    """Return ``(values, method_flag)`` for a vital sign.

    Sentinels and documented non-measurement codes become ``NaN``. The method flag
    is 1 exactly where a documented method code was used, so that the
    information "measured by Doppler" survives instead of being thrown away with
    the value.
    """
    numeric = _to_numeric(series)
    flag = pd.Series(0, index=numeric.index, dtype="int8")

    if method_code is not None:
        is_method = numeric == method_code
        flag = flag.mask(is_method, 1).astype("int8")
        numeric = numeric.mask(is_method)

    if unobtainable_code is not None:
        numeric = numeric.mask(numeric == unobtainable_code)

    numeric = numeric.mask(numeric.isin([BLANK, UNKNOWN]))
    return numeric, flag


# --- The main cleaning entry point -------------------------------------------


def build_modelling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Project a raw NHAMCS frame onto the Part 1 modelling features.

    Row-local and stateless. Adds no information and removes no rows.
    """
    out = pd.DataFrame(index=frame.index)

    # -- Numeric measurement block --------------------------------------------
    out[ARRIVAL_MINUTES] = parse_arrival_time(frame["ARRTIME"])

    out[TEMP_F] = _blank_to_nan(frame["TEMPF"]) / TEMPF_SCALE_DIVISOR

    pulse_values, pulse_flag = _measurement_with_method_flag(
        frame["PULSE"], PULSE_METHOD_CODE
    )
    out[PULSE_BPM] = pulse_values
    out[PULSE_METHOD] = pulse_flag

    out[RESP_RATE] = _blank_to_nan(frame["RESPR"])

    out[BP_SYSTOLIC] = _measurement_with_method_flag(
        frame["BPSYS"], None, unobtainable_code=BPSYS_UNOBTAINABLE_CODE
    )[0]

    diastolic_values, diastolic_flag = _measurement_with_method_flag(
        frame["BPDIAS"], BPDIAS_METHOD_CODE
    )
    out[BP_DIASTOLIC] = diastolic_values
    out[BP_DIASTOLIC_METHOD] = diastolic_flag

    out[SPO2_PCT] = _blank_to_nan(frame["POPCT"])

    # -- Categorical response block -------------------------------------------
    out[VISIT_MONTH] = _to_numeric(frame["VMONTH"]).astype("Int64").astype("string")
    out[VISIT_DOW] = _to_numeric(frame["VDAYR"]).astype("Int64").astype("string")

    out[PATIENT_RESIDENCE] = _categorical_with_labels(frame["RESIDNCE"])
    out[ARRIVAL_BY_AMBULANCE] = _categorical_with_labels(frame["ARREMS"])
    # -7 is structural here: the item only applies to ambulance arrivals.
    out[INTERFACILITY_TRANSFER] = _categorical_with_labels(
        frame["AMBTRANSFER"],
        structural_absent=NOT_APPLICABLE,
        absent_label=SENTINEL_LABELS[NOT_APPLICABLE],
    )
    out[SEEN_WITHIN_72H] = _categorical_with_labels(frame["SEEN72"])

    # -- High-cardinality reason-for-visit block -------------------------------
    # The 4-digit forms only: eddict2022.dct places RFV1 at bytes 73-77 and
    # RFV13D at 73-76, i.e. a genuine 4-character truncation. The dropped 5th
    # digit encodes symptom-character detail (doc p.208) not necessarily elicited
    # before triage. 0 is structural absence.
    for raw_column, engineered in zip(
        ["RFV13D", "RFV23D", "RFV33D", "RFV43D", "RFV53D"], RFV_FEATURES
    ):
        out[engineered] = _categorical_with_labels(
            frame[raw_column],
            structural_absent=RFV_ABSENT_CODE,
            absent_label=RFV_ABSENT_LABEL,
        )

    return out


# --- Phase 5: range validation (report only, never modifies) -----------------


def validate_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    """Check engineered numeric features against their documented ranges.

    Returns a report. **No value is clipped, winsorised or removed.** Violations
    are surfaced so they can be discussed, which is the required treatment for
    NHAMCS: the apparent anomalies are documented meanings, and observation-based
    outlier deletion would remove real clinical signal.
    """
    rows = []
    total = len(frame)
    for column, (low, high, note) in DOCUMENTED_RANGES.items():
        values = frame[column]
        observed = values.dropna()
        below = int((observed < low).sum())
        above = int((observed > high).sum())
        rows.append(
            {
                "feature": column,
                "documented_lo": low,
                "documented_hi": high,
                "n_observed": int(observed.shape[0]),
                "n_missing": int(total - observed.shape[0]),
                "n_below": below,
                "n_above": above,
                "obs_min": float(observed.min()) if observed.shape[0] else np.nan,
                "obs_median": round(float(observed.median()), 2) if observed.shape[0] else np.nan,
                "obs_max": float(observed.max()) if observed.shape[0] else np.nan,
                "in_documented_range": below == 0 and above == 0,
                "source": note,
            }
        )
    return pd.DataFrame(rows)


def flagged_implausible_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count in-range but clinically implausible values. Reported, not altered."""
    rows = []
    for column, codes in FLAGGED_IMPLAUSIBLE.items():
        for code, note in codes.items():
            count = int((frame[column] == code).sum())
            if count:
                rows.append({"feature": column, "value": code, "n": count, "note": note})
    return pd.DataFrame(rows)


def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Missingness per engineered feature. Descriptive only."""
    total = len(frame)
    rows = []
    for column in NUMERIC_FEATURES:
        rows.append(
            {
                "feature": column,
                "kind": "numeric",
                "n_missing": int(frame[column].isna().sum()),
                "pct_missing": round(100.0 * frame[column].isna().sum() / total, 2),
                "n_levels": np.nan,
            }
        )
    for column in CATEGORICAL_FEATURES:
        rows.append(
            {
                "feature": column,
                "kind": "categorical",
                # Recorded response states are categories, not missingness.
                "n_missing": int((frame[column] == "missing").sum()),
                "pct_missing": round(100.0 * (frame[column] == "missing").sum() / total, 2),
                "n_levels": int(frame[column].nunique()),
            }
        )
    return pd.DataFrame(rows)
