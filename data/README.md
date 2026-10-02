# Data

## Status: intentionally empty

There are no data files in this directory, and none are coming.

The project uses the **2022 NHAMCS (National Hospital Ambulatory Medical Care
Survey) Emergency Department public-use file**. The raw file is **not**
redistributed here. See *Why* below.

## Why the raw file is not in this repository

1. **Size.** `ed2022-stata.dta` is 37,720,094 bytes (≈35.97 MiB). Committing it
   would put a large binary blob in Git history permanently — it cannot be
   cleanly removed later without a history rewrite, which this project does not do.
2. **Licensing is unresolved.** The official `readme2022.txt` places the data
   under the Public Health Service Act §308(d) (42 U.S.C. 242m(d)) and CIPSEA
   (44 U.S.C. 3561–3583). Those terms govern *use*; they do not state whether
   the file may be re-hosted. Redistributing it here would assert a permission
   that has not been confirmed with NCHS.
3. **The course brief requires a GitHub link from Part 1 onward.** That
   requirement is satisfied by publishing code, documentation and derived
   aggregate tables — not by re-hosting someone else's data file.

`.gitignore` blocks `data/raw/`, `*.dta`, `*.zip` and related statistical-data
extensions, so the raw file cannot be committed by accident.

## How to obtain it

Official source, as recorded in the forensic audit's
`verification/verification_report.txt`:

- Landing page — `https://www.cdc.gov/nchs/nhamcs/documentation/`
- Archive — `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ed2022-stata.zip`

These URLs are transcribed from the audit's acquisition record. They were not
re-fetched during repository initialisation; re-verify against the current CDC
page before relying on them.

Extract the archive to a location **outside** this repository, e.g.:

```
C:\Projects\ML_Dataset_Audit\NHAMCS_2022\raw\nhamcs_ed_2022\ed2022-stata.dta
```

## Expected dataset identity

Verify the file before using it. Any mismatch means the wrong file, and the
analysis must not proceed.

| Property | Expected value |
|---|---|
| Filename | `ed2022-stata.dta` |
| Size | `37,720,094` bytes |
| SHA-256 | `0E00CD5AEA4511CD619F7AF1E0DC2F6825F111B7C87D3EBBB1358410A4C881C2` |
| Shape | `16,025` rows × `913` columns |
| Target column | `IMMEDR` |
| Archive (optional cross-check) | `ed2022-stata.zip`, SHA-256 `AA32E635EF9564B1C8E03CEA16C87EC2ED02E9A928D22A257DF18100A38AD683` |

These values were confirmed against the file on 2026-10-02, both during repository
initialisation and again during the Part 1 implementation run.

Windows check:

```powershell
Get-FileHash -LiteralPath "<path>\ed2022-stata.dta" -Algorithm SHA256
(Get-Item -LiteralPath "<path>\ed2022-stata.dta").Length
```

> **Note.** Read the file with `pandas.read_stata(..., convert_categoricals=False)`.
> The default categorical conversion raises `ValueError` on this official file
> because the `TEMPF` value-label set contains a duplicate label. That is a
> pandas converter limitation, not a file defect.

## Configurable local path — implemented

The code reads the dataset path from configuration. **No absolute path is
hard-coded**; every path in `src/config.py` is resolved relative to the file, so
the repository can be cloned anywhere. `resolve_dataset_path()` tries, in order:

1. an **explicit argument**, if one is passed — checked first, and if it does not
   exist the call raises rather than silently falling back to another candidate;
2. the **`NHAMCS_DATA_PATH`** environment variable;
3. `data/local/ed2022-stata.dta` — a git-ignored drop location;
4. the sibling forensic-audit workspace,
   `../ML_Dataset_Audit/NHAMCS_2022/raw/nhamcs_ed_2022/ed2022-stata.dta`,
   addressed *relative* to this repository's parent.

On failure it raises `FileNotFoundError` listing every path it searched.

### The integrity gate is enforced, not advisory

`src/data/dataset.py` recomputes the size and SHA-256 and **refuses to load**
unless both match the pinned values, then confirms the shape after parsing:

```python
from src.data import dataset

report = dataset.verify_dataset()          # size + SHA-256 only
full = dataset.load_raw_dataset()          # verifies, then loads; asserts shape
```

The digest is the gate rather than the shape, because a digest match establishes
byte-identity and therefore identity of shape as well.

## Reproducibility

Running `notebooks/01_assignment1_part1_eda_and_pipeline.ipynb` top to bottom
reproduces every figure and every table in `docs/`. The notebook is committed
with its outputs and all four figures already populated, so it can be read
without executing anything. The code that produced those figures lives in
`src/`; the notebook imports it rather than redefining it.

The forensic audit that established the dataset's properties is documented in
[`../audit/README.md`](../audit/README.md).

## Nothing derived is committed either

Machine-generated binaries (`*.parquet`, `*.feather`, `*.pkl`, `*.joblib`) and
scratch output directories are git-ignored. Aggregate tables that support the
written report are added deliberately, one commit at a time.
