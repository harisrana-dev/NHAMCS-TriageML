# Environment — authoritative specification

> Single source of truth for the NHAMCS-TriageML runtime environment
> (Issue #23). When another document records a different environment, the
> record there is *provenance for results produced at that time*; this file
> defines what to install to work on the project now.

## Supported Python

| Item | Value |
|---|---|
| Supported version | **CPython 3.13.x** |
| Reference / tested | **CPython 3.13.5**, Windows 11 x64 |
| Check | `python -V` → `Python 3.13.x` |

3.13.5 is the patch release every committed result was verified on — it is the
**tested reference**, not the only accepted patch release. Other 3.13.x patch
releases should work; re-run the verification commands below after changing one.

Earlier results carry their own historical environments, kept as provenance:
`reports/decision_tree_gini.md` (CPython 3.12.10) and
`docs/gaussian_nb_experiment.md` (Python 3.14.2). Those are records of what ran
then, not setup instructions.

## Dependency manifests

| File | Scope |
|---|---|
| `requirements.txt` | **Core** — data loading, preprocessing, evaluation framework, PCA / K-means / sklearn-native interpretability, notebook execution, tests. Everyone installs this. |
| `requirements-xgboost.txt` | Optional — XGBoost (Issue #13). |
| `requirements-torch-cpu.txt` | Optional — PyTorch, CPU-only wheels from the official PyTorch index (Issue #16). |

Rules:

- `requirements.txt` is the only core manifest; optional model libraries never
  belong in it (Issue #23).
- `scikit-learn==1.9.1` is **exactly pinned** in the core manifest. Fold
  assignments from `StratifiedGroupKFold` can differ between scikit-learn
  versions even with fixed seeds, so Part 2/3 results are only comparable under
  the pin — see `docs/decision_tree_results.md` §10.
- The PyTorch package index is declared **only** inside
  `requirements-torch-cpu.txt`; it must never appear in the core or XGBoost
  manifests.
- Interpretability uses **scikit-learn-native** tools
  (`sklearn.inspection.permutation_importance`, etc.). SHAP and LIME are
  deliberately not dependencies yet.

## Installation

Windows PowerShell, from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt          # core — everyone
```

Optional model work, on top of the core:

```powershell
python -m pip install -r requirements-xgboost.txt  # XGBoost (Issue #13)
python -m pip install -r requirements-torch-cpu.txt # PyTorch CPU-only (Issue #16)
```

Install an optional manifest only when you need that model — not by default.
`macOS`/`Linux` use `source .venv/bin/activate` for the virtual environment;
everything else is identical.

## Dataset path precedence

The raw `ed2022-stata.dta` is **not** in this repository (see
`data/README.md`). `src/config.resolve_dataset_path()` searches, in order:

1. an **explicit argument** — checked first; if given but missing, the call
   raises instead of falling back;
2. the **`NHAMCS_DATA_PATH`** environment variable;
3. **`data/local/ed2022-stata.dta`** — the git-ignored drop location (the
   usual place to put the file);
4. the sibling forensic-audit workspace,
   `../ML_Dataset_Audit/NHAMCS_2022/raw/nhamcs_ed_2022/ed2022-stata.dta`,
   addressed relative to this repository's parent.

On failure it raises `FileNotFoundError` listing every path tried. Nothing
hard-codes an absolute path.

## Verification commands

Run from the repository root:

```bash
python -V                          # Python 3.13.x
python -m pip check                # -> No broken requirements found.
python -m pytest -q                # 69 tests: 64 unit + 5 integration
python -m pytest -m "not integration" -q   # 64 unit tests (no dataset needed)
python -m pytest -m integration -q         # 5 dataset-backed tests
```

- **Without the raw dataset** the full suite still runs: the 5 integration
  tests skip themselves (`tests/test_integration_real_data.py`) and the 64
  unit tests pass. This is what a fresh clone sees.
- **With the dataset discoverable** all 69 pass, including
  `test_part2_baseline_metrics_and_figure_unchanged`, which asserts the Part 2
  headline metrics and the SHA-256 of
  `reports/figures/fig05_part2_macro_f1_comparison.png`.
- Optional smoke checks:

```bash
python -c "import xgboost; print(xgboost.__version__)"   # 3.4.1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 2.15.0+cpu False
```

## Reference environment record

Verified in a clean `py -3.13` virtual environment during Issue #23:

| Package | Version |
|---|---|
| Python | 3.13.5 |
| pandas | 2.3.3 (floor `>=2.3.3`) |
| numpy | 2.5.5 (floor `>=2.5.4`) |
| scikit-learn | **1.9.1 (pinned)** |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| jupyterlab | 4.6.6 |
| ipykernel | 7.4.0 |
| nbformat | 5.11.1 |
| nbclient | 0.11.0 |
| pytest | 9.2.0 (floor `>=9.1.1`) |
| xgboost (optional) | 3.4.1 |
| torch (optional, CPU) | 2.15.0+cpu |

The environment that produced the committed results recorded numpy 2.5.4,
jupyterlab 4.6.4 and pytest 9.1.1 (see the header of `requirements.txt`); the
clean install above resolved slightly newer versions *inside the same ranges*,
which is exactly what ranges without a lock file do.

`pip check` reports no broken requirements in the core, XGBoost and PyTorch
environments. XGBoost fits a synthetic 5-class task alongside scikit-learn
1.9.1; the CPU wheel of PyTorch trains a minimal classifier with
`torch.cuda.is_available() == False`.

## Portability limitations

- **No lock file and no CI** — by decision (Issue #23): ranges plus the exact
  `scikit-learn` pin are the reproducibility contract; CI will be added only if
  the team can maintain it.
- **Exact pins only for scikit-learn.** Other packages resolve within their
  ranges, so a later install may pick a newer in-range version than the record
  above.
- **Torch wheels are platform-specific.** The manifest resolves CPU builds on
  Windows/macOS/Linux x64 via the PyTorch CPU index; `+cpu` is required for the
  local version segment.
- **Windows 11 x64 is the tested platform.** Other platforms are expected to
  work but are unverified here.
