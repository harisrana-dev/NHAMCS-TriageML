"""Final Part 1 feature set, the leakage-free ColumnTransformer, and leakage guards.

Feature-selection policy
------------------------
The forensic audit classified all 913 variables of the file
(``audit/variable_inventory.csv``). Part 1 adopts a deliberately narrow subset:

* **Class A (13)** - timing-eligible. All included.
* **Class B, reason-for-visit** - the 4-digit ``RFV13D``-``RFV53D`` forms only.
  Included under an explicitly stated assumption (see :data:`RFV_ASSUMPTION`).
* **Class B, everything else** - payment (10), ``PAINSCALE``, ``EPISODE``.
  **Excluded.** The audit tags their collection point relative to the triage
  decision as UNRESOLVED, i.e. the documentation is silent. An assumption cannot
  be stated for them, so they are not guessed at.
* **Class G (24)** - excluded. Timing unresolved by the audit.
* **Demographics** ``AGE``, ``SEX``, ``RACER``, ``RACERETH``, ``ETHIM``,
  ``ETHUN``, ``RACEUN`` - **excluded**. The audit classes these as C (LEAKAGE),
  not E: NCHS imputed them with a model whose inputs included triage level.
* **Class E** - keys and survey design. Never features.
* **Class F (41)** - facility-only constants. Never features.
* **Class C + D (799)** - recorded after the triage decision, or derived from
  such fields. Never features.

Hospital identifiers are excluded by construction and enforced by
:func:`assert_no_leakage`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import sentinels as sn

# --- Leakage prohibitions -----------------------------------------------------

TARGET = "IMMEDR"

#: Keys and survey-design variables. Never features. HOSPCODE additionally
#: defines the grouping structure for the later grouped-validation stage, but it
#: must not enter the predictive matrix.
HOSPITAL_IDENTIFIERS: frozenset[str] = frozenset(
    {"HOSPCODE", "PATCODE", "CPSUM", "CSTRATM"}
)

#: Demographics carrying NCHS imputation that used the target as an input.
DEMOGRAPHIC_TARGET_IMPUTED: frozenset[str] = frozenset(
    {"AGE", "SEX", "RACER", "RACERETH", "RACEUN", "ETHIM", "ETHUN"}
)

#: The stated assumption that makes RFV usable. Quoted verbatim in the notebook
#: and in docs/feature_register.md so it cannot be quietly dropped.
RFV_ASSUMPTION = (
    "The patient's presenting complaint is stated before the triage decision, and "
    "the released RFV code is a faithful abstraction of that complaint. Verified "
    "counter-evidence: RTI assigned the released code retrospectively from the "
    "verbatim complaint (doc22-ed-508.pdf p.11), with a documented 2.4% overall "
    "recoding error rate. The code, not the complaint, is what enters the model."
)

#: Minimum training-fold frequency for a reason-for-visit category before it is
#: folded into a shared infrequent bucket. Fitted INSIDE the pipeline on training
#: data only. The forensic audit recommends 10, on the grounds that rare RFV codes
#: are disproportionately high-acuity and must be grouped, not dropped.
RFV_MIN_FREQUENCY = 10


# --- The final feature sets ---------------------------------------------------

NUMERIC: tuple[str, ...] = sn.NUMERIC_FEATURES
BINARY_FLAGS: tuple[str, ...] = sn.METHOD_FLAG_FEATURES
CATEGORICAL_LOW: tuple[str, ...] = sn.LOW_CARDINALITY_CATEGORICAL
CATEGORICAL_HIGH: tuple[str, ...] = sn.RFV_FEATURES
FEATURES: tuple[str, ...] = sn.MODELLING_FEATURES

#: The 18 raw source variables the register draws on, in codebook order. Declared
#: explicitly rather than derived by string surgery, so the "18 raw -> 20
#: engineered" claim in the feature register is a checkable fact. ``PULSE`` and
#: ``BPDIAS`` each back two engineered columns (a value and a method flag), which
#: is why the counts differ.
RAW_SOURCE_VARIABLES: tuple[str, ...] = (
    "ARRTIME",
    "TEMPF",
    "PULSE",
    "RESPR",
    "BPSYS",
    "BPDIAS",
    "POPCT",
    "VMONTH",
    "VDAYR",
    "RESIDNCE",
    "ARREMS",
    "AMBTRANSFER",
    "SEEN72",
    "RFV13D",
    "RFV23D",
    "RFV33D",
    "RFV43D",
    "RFV53D",
)


# --- Leakage guards -----------------------------------------------------------


def leakage_checks(features: tuple[str, ...] = FEATURES) -> pd.DataFrame:
    """Assert and report leakage checks for the proposed feature list.

    Raises ``AssertionError`` on any violation. Called at pipeline-construction
    time and again in the notebook, so a future edit cannot quietly reintroduce a
    prohibited variable.
    """
    features_set = set(features)
    problems: list[str] = []

    if TARGET in features_set:
        problems.append(f"The target {TARGET!r} is present in the feature list.")

    leaked_ids = sorted(features_set & HOSPITAL_IDENTIFIERS)
    if leaked_ids:
        problems.append(f"Hospital/survey identifiers present: {leaked_ids}")

    leaked_demo = sorted(features_set & DEMOGRAPHIC_TARGET_IMPUTED)
    if leaked_demo:
        problems.append(
            f"Demographics with NCHS target-linked imputation present: {leaked_demo}"
        )

    unknown = sorted(features_set - set(sn.MODELLING_FEATURES))
    if unknown:
        problems.append(f"Unrecognised columns in the feature list: {unknown}")

    if problems:
        raise AssertionError("Leakage guard failed:\n  - " + "\n  - ".join(problems))

    return pd.DataFrame(
        [
            {
                "check": "target absent from feature list",
                "detail": f"{TARGET!r} not in the {len(features_set)} features",
                "result": "PASS",
            },
            {
                "check": "hospital / survey identifiers absent",
                "detail": "none of " + ", ".join(sorted(HOSPITAL_IDENTIFIERS)),
                "result": "PASS",
            },
            {
                "check": "target-imputed demographics absent",
                "detail": "none of "
                + ", ".join(sorted(DEMOGRAPHIC_TARGET_IMPUTED)),
                "result": "PASS",
            },
            {
                "check": "no unrecognised columns",
                "detail": f"all {len(features_set)} map to the documented register",
                "result": "PASS",
            },
            {
                "check": "register draws on 18 raw variables",
                "detail": f"{len(RAW_SOURCE_VARIABLES)} declared raw source variables",
                "result": "PASS" if len(RAW_SOURCE_VARIABLES) == 18 else "FAIL",
            },
        ]
    )


def eligibility_class_summary() -> pd.DataFrame:
    """Audit class counts for the whole 913-variable file, and the Part 1 decision."""
    return pd.DataFrame(
        [
            {"class": "A", "meaning": "Timing-eligible", "n": 13,
             "part1_decision": "INCLUDED"},
            {"class": "B (RFV)", "meaning": "Conditional: reason for visit", "n": 10,
             "part1_decision": "INCLUDED (4-digit forms only, assumption stated)"},
            {"class": "B (other)", "meaning": "Conditional: payment, PAINSCALE, EPISODE",
             "n": 12, "part1_decision": "EXCLUDED (collection point unresolved)"},
            {"class": "C", "meaning": "Recorded after the triage decision", "n": 794,
             "part1_decision": "EXCLUDED"},
            {"class": "D", "meaning": "Derived from class C", "n": 5,
             "part1_decision": "EXCLUDED"},
            {"class": "E", "meaning": "Key / admin / survey design", "n": 13,
             "part1_decision": "EXCLUDED"},
            {"class": "F", "meaning": "Facility-only (constant within HOSPCODE)", "n": 41,
             "part1_decision": "EXCLUDED"},
            {"class": "G", "meaning": "Timing unresolved", "n": 24,
             "part1_decision": "EXCLUDED"},
            {"class": "T", "meaning": "Target (IMMEDR)", "n": 1,
             "part1_decision": "TARGET"},
        ]
    )


# --- The pipeline -------------------------------------------------------------


def _select_block_columns(
    block_columns: Sequence[str], selected: frozenset[str]
) -> list[str]:
    """Keep only register columns that remain in ``selected``, preserving order."""
    return [column for column in block_columns if column in selected]


def build_column_transformer(
    columns: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Construct the three-block ColumnTransformer for the feature set.

    Parameters
    ----------
    columns:
        Optional subset of :data:`FEATURES`. When ``None`` (the default), the
        full approved register is used and Part 1 behaviour is unchanged.
        A non-``None`` value must be a leakage-safe subset of the register;
        :func:`leakage_checks` still runs on the selected columns. Blocks whose
        columns are all removed remain present as empty transformers so the
        output schema stays block-structured.

    Every learned parameter - imputation statistics, category vocabulary, rare
    category thresholds, scaling moments - is fitted inside ``Pipeline.fit`` on the
    training fold only. Nothing is precomputed on the full dataset.

    Anything outside the register is dropped by ``remainder="drop"``. The modelling
    frame is 20 engineered columns derived from 18 raw source variables, so the
    other 895 of the file's 913 columns - the target included - can never reach the
    matrix. Publishing that register is itself a Part 1 deliverable, not an
    implementation detail.
    """
    if columns is None:
        selected_features: tuple[str, ...] = FEATURES
    else:
        selected_features = tuple(columns)
    leakage_checks(selected_features)
    selected = frozenset(selected_features)

    numeric_columns = _select_block_columns(NUMERIC, selected)
    flag_columns = _select_block_columns(BINARY_FLAGS, selected)
    low_cardinality_columns = _select_block_columns(CATEGORICAL_LOW, selected)
    high_cardinality_columns = _select_block_columns(CATEGORICAL_HIGH, selected)

    numeric_block = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    # The forensic audit establishes that vital-sign missingness is
                    # associated with the target (Cramer's V 0.066-0.137), so the
                    # pattern itself carries information and is retained rather
                    # than being absorbed into the imputed value.
                    add_indicator=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    # A constant imputer, not a most-frequent one. Recorded response states
    # ("blank", "unknown", "not_applicable", "none_recorded") are explicit
    # categories, so on the current data this step is a no-op. It is retained to
    # guarantee a total function if a fold ever yields a genuinely absent cell,
    # and it deliberately never fabricates a modal response.
    low_cardinality_block = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="__absent__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    high_cardinality_block = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="__absent__")),
            (
                "onehot",
                OneHotEncoder(
                    # Grouped inside the pipeline, from training-fold counts only.
                    # Categories rarer than the threshold collapse into one shared
                    # bucket per slot rather than being dropped, because rare
                    # complaint categories are disproportionately high-acuity and
                    # must not silently vanish.
                    min_frequency=RFV_MIN_FREQUENCY,
                    # Required, not cosmetic: grouped validation holds out whole
                    # hospitals, so every evaluation fold contains complaint
                    # categories absent from training. "infrequent_if_exist" maps
                    # both those unseen codes and the rare ones into the single
                    # infrequent column. Plain "ignore" would instead emit an
                    # all-zero row, which a linear model reads as "nothing was
                    # observed" - indistinguishable from an unremarkable visit.
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_block, numeric_columns),
            # Binary method indicators pass through unscaled. They are not
            # measurements: standardising a 0/1 would rescale it to a
            # z-score and destroy its reading as a flag. They also must never be
            # imputed - a flag is only meaningful as observed. Note these columns
            # would otherwise be silently dropped by remainder="drop", which is
            # how the Doppler/palpated provenance of PULSE=998 and BPDIAS=998
            # would be lost.
            ("method_flags", "passthrough", flag_columns),
            ("categorical_low", low_cardinality_block, low_cardinality_columns),
            ("categorical_high", high_cardinality_block, high_cardinality_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def preprocessing_only(columns: Sequence[str] | None = None) -> Pipeline:
    """The fitted-able preprocessing stage, as a standalone Pipeline."""
    return Pipeline(steps=[("preprocess", build_column_transformer(columns))])


def build_pipeline(
    estimator: BaseEstimator | str | None = None,
    columns: Sequence[str] | None = None,
) -> Pipeline:
    """Return the full Part 1 Pipeline, optionally with a live estimator.

    Parameters
    ----------
    estimator:
        When ``None`` (the default), the final step keeps the Part 1
        ``"placeholder"`` string so the scaffold stays inspectable without a
        classifier. Pass a live estimator instance to obtain a fittable
        Part 2 pipeline. Each caller should pass a **fresh** estimator.
    columns:
        Optional feature subset forwarded to
        :func:`build_column_transformer`. ``None`` keeps the full register.

    The final estimator is left as a named placeholder by default. Part 1's
    deliverable is the leakage-free preprocessing design; a classifier belongs
    to Part 2. The placeholder keeps the object a genuine ``Pipeline`` so its
    structure and fit semantics can be inspected and tested now, and so Part 2
    can attach an estimator without restructuring anything.
    """
    final_step: BaseEstimator | str = (
        "placeholder" if estimator is None else estimator
    )
    return Pipeline(
        steps=[
            ("preprocess", build_column_transformer(columns)),
            ("estimator", final_step),
        ]
    )
