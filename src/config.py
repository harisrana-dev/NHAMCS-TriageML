"""Project-wide configuration: repository paths and verified dataset identity.

The raw NHAMCS file is deliberately absent from this repository (see
``data/README.md``). Nothing here hard-codes a machine-specific absolute path.
The dataset is located at run time through :func:`resolve_dataset_path`, in this
order:

1. the ``NHAMCS_DATA_PATH`` environment variable, if set;
2. ``<repo>/data/local/ed2022-stata.dta`` (git-ignored drop location);
3. the sibling forensic-audit workspace, addressed *relative* to this
   repository's parent directory.

All paths are resolved relative to this file, so the repository can be cloned
anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# --- Verified dataset identity ------------------------------------------------
# Source of record: the forensic audit workspace
# (ML_Dataset_Audit/NHAMCS_2022/verification/verification_report.txt).
# Re-verified against the file on 2026-10-02 during Part 1 implementation.

DATASET_FILENAME = "ed2022-stata.dta"
EXPECTED_SHA256 = "0E00CD5AEA4511CD619F7AF1E0DC2F6825F111B7C87D3EBBB1358410A4C881C2"
EXPECTED_SIZE_BYTES = 37_720_094
EXPECTED_SHAPE = (16_025, 913)

# The Sentry-label duplicate that makes pandas' default converter fail on this
# official file. Read with convert_categoricals=False; see data/README.md.
PANDAS_READ_KWARGS = {"convert_categoricals": False}

# --- Reproducibility ----------------------------------------------------------
# Single source of truth for every random seed used in Part 1, so that the
# demonstration split in the notebook and any later stage agree.
RANDOM_STATE = 42

# Environment variable consulted first by resolve_dataset_path().
DATA_PATH_ENV_VAR = "NHAMCS_DATA_PATH"


def _candidate_paths() -> list[Path]:
    """Return candidate dataset locations in resolution priority order."""
    candidates: list[Path] = []

    from_env = os.environ.get(DATA_PATH_ENV_VAR)
    if from_env:
        candidates.append(Path(from_env).expanduser())

    candidates.append(DATA_DIR / "local" / DATASET_FILENAME)

    # Sibling forensic-audit workspace, addressed relatively so the repository
    # stays portable. C:\Projects\NHAMCS-TriageML -> ../ML_Dataset_Audit/...
    candidates.append(
        REPO_ROOT.parent
        / "ML_Dataset_Audit"
        / "NHAMCS_2022"
        / "raw"
        / "nhamcs_ed_2022"
        / DATASET_FILENAME
    )
    return candidates


def resolve_dataset_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate the raw ``.dta`` file, or raise with an actionable message.

    Parameters
    ----------
    explicit:
        Optional explicit path, **checked first**. If it is given and exists, it
        wins over every other candidate.
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"An explicit dataset path was supplied but does not exist:\n  {candidate}\n"
            "Correct the path, or call resolve_dataset_path() with no argument to "
            f"fall back to {DATA_PATH_ENV_VAR} and the standard locations."
        )

    searched: list[Path] = []
    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate
        searched.append(candidate)

    tried = "\n  ".join(str(p) for p in searched)
    raise FileNotFoundError(
        "Could not locate the NHAMCS raw dataset.\n"
        "It is intentionally not committed to this repository. Obtain\n"
        f"  {DATASET_FILENAME}\n"
        "separately from the official CDC/NCHS source (see data/README.md),\n"
        "then either:\n"
        f"  - set the {DATA_PATH_ENV_VAR} environment variable, or\n"
        f"  - place the file at {DATA_DIR / 'local' / DATASET_FILENAME}\n"
        "Paths searched:\n  " + tried
    )
