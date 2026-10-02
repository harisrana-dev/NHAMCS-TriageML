"""Dataset loading, integrity verification and target/population definition."""

from src.data.dataset import load_raw_dataset, sha256_of, verify_dataset
from src.data.target import (
    NON_URGENCY_CODES,
    TARGET_COLUMN,
    URGENCY_CLASSES,
    URGENCY_LABELS,
    exclusion_summary,
    is_modelling_population,
    majority_class_baseline,
    select_modelling_population,
    target_summary,
)

__all__ = [
    "load_raw_dataset",
    "verify_dataset",
    "sha256_of",
    "TARGET_COLUMN",
    "URGENCY_CLASSES",
    "URGENCY_LABELS",
    "NON_URGENCY_CODES",
    "is_modelling_population",
    "select_modelling_population",
    "target_summary",
    "exclusion_summary",
    "majority_class_baseline",
]
