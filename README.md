# NHAMCS-TriageML

Machine-learning study of emergency-department triage urgency using the 2022
NHAMCS Emergency Department public-use dataset.

> **Status: Assignment 1, Part 1 implemented and verified. No model trained.**
>
> The official dataset is loaded and verified byte-for-byte, the target and the
> final feature register are settled, and the leakage-free preprocessing pipeline
> is implemented and executed. **No classifier has been fitted, no accuracy or
> macro-F1 figure has been produced, and none is claimed.** Instructor approval of
> the dataset is still outstanding.

## Research question

> How accurately can five-class ED triage acuity (`IMMEDR` 1–5) be predicted at
> arrival from timing, vital signs, and documented reason-for-visit information
> in the 2022 NHAMCS ED dataset, and how much of that performance survives when
> evaluation holds out whole hospitals?

The second clause is the point of the project. The file spans **147 hospitals**
that differ in triage policy, and hospital identity alone reaches Cramer's V
0.303 against the target. A random row split measures memorisation as much as
generalisation.

## Target

`IMMEDR` — "immediacy with which patient should be seen (unimputed)" — as a
**five-class ordinal** problem. Primary metric: **macro-averaged F1**.

| Code | Level | n | % |
|---|---|---|---|
| 1 | Immediate | 139 | 1.36 |
| 2 | Emergent | 1,595 | 15.63 |
| 3 | Urgent | 5,340 | 52.32 |
| 4 | Semi-urgent | 2,826 | 27.69 |
| 5 | Non-urgent | 307 | 3.01 |

Modelling population: **10,207 of 16,025 visits (63.68%)**. The 5,818 exclusions
are documented, not imputed — and include codes `0` and `7`, which are
*documentation states*, not urgency levels. Majority-class baseline: **0.5232**.

## Dataset

**2022 NHAMCS Emergency Department public-use dataset**
(National Hospital Ambulatory Medical Care Survey, CDC/NCHS).

| Property | Value |
|---|---|
| File | `ed2022-stata.dta` |
| Size | 37,720,094 bytes |
| SHA-256 | `0E00CD5AEA4511CD619F7AF1E0DC2F6825F111B7C87D3EBBB1358410A4C881C2` |
| Shape | 16,025 rows × 913 columns |
| Target column | `IMMEDR` |

The raw file is **not redistributed here**. It must be obtained separately from
the official source; see [`data/README.md`](data/README.md) for the URL and the
hash check. The loader refuses to proceed unless size **and** SHA-256 match.

**NHAMCS, not NHCS.** An earlier acquisition retrieved `nhcs2022ed_stata.dta`
(NHCS — National Hospital Care Survey, 497,769 rows). It is quarantined in the
audit workspace and must never be used. See [`audit/README.md`](audit/README.md).

## Features

**18 raw source variables → 20 engineered columns**, selected by *collection
timing* rather than by correlation with the target. Full table, sentinel rules and
level counts: [`docs/feature_register.md`](docs/feature_register.md).

| Block | Columns | Raw variables |
|---|---|---|
| Numeric measurements | 7 | `ARRTIME` `TEMPF` `PULSE` `RESPR` `BPSYS` `BPDIAS` `POPCT` |
| Binary method flags | 2 | `PULSE` `BPDIAS` (Doppler / palpated provenance) |
| Low-cardinality categorical | 6 | `VMONTH` `VDAYR` `RESIDNCE` `ARREMS` `AMBTRANSFER` `SEEN72` |
| Reason-for-visit, 4-digit | 5 | `RFV13D` `RFV23D` `RFV33D` `RFV43D` `RFV53D` |

## Findings that constrain the project

From the forensic audit. These are findings, not problems to be worked around.

- **913 variables, 35 candidates.** Only 13 are unambiguously available at or
  before the triage decision. 794 are recorded after it and 5 derive from those.
- **Circular imputation.** NCHS imputed `AGE`, `SEX`, `RACER`, `RACERETH`,
  `RACEUN`, `ETHIM`, `ETHUN` using a model whose **inputs included triage level**.
  **22.84%** of the modelling population carries at least one such value. These
  variables are therefore **excluded**: they are leakage, not merely "recorded
  later".
- **Validation.** Records are clustered in **147 hospitals**, and facility
  identity is associated with the target. A random row split is invalid;
  grouped validation on `HOSPCODE` is required.
- **Reason-for-visit is an assumption.** The 4-digit RFV codes are retained under
  a disclosed assumption, with documented counter-evidence: RTI assigned the
  released code retrospectively, with a 2.4% recoding error rate. RFV reaches
  Cramer's V 0.41–0.51 — the strongest signal in the file — so the audit calls
  this the single most consequential judgement in the project.
- **Honest signal is weak.** The six initial vital signs — the only
  unambiguously pre-triage physiological measurements — are essentially
  unassociated with the assigned triage level.

## Project

**University of Lahore**
Faculty of Information Technology — Department of Computer Science & IT
**CS-13410 — Introduction to Machine Learning**
Fall 2026, BSCS

## Repository structure

```
NHAMCS-TriageML/
├── README.md              # this file
├── requirements.txt       # dependencies, with the versions observed locally
├── data/README.md         # how to obtain and verify the dataset — no data files
├── notebooks/
│   └── 01_assignment1_part1_eda_and_pipeline.ipynb   # executed; 4 figures inline
├── src/
│   ├── config.py          # dataset resolution, pinned identity, seed
│   ├── data/              # integrity verification + target definition
│   ├── preprocessing/     # sentinels, feature register, ColumnTransformer
│   └── evaluation/        # reserved for Part 2 (grouped CV, metrics)
├── docs/
│   ├── assignment1_proposal.md   # the one-page proposal
│   ├── feature_register.md       # final 18 raw -> 20 engineered
│   └── methodology.md            # why each decision was made
├── reports/figures/       # committed figures backing the report
└── audit/README.md        # pointer to the external forensic audit workspace
```

Division of responsibility:

- `src/` — deterministic, testable code. No notebook-only logic.
- `notebooks/` — the narrative deliverable, executed end-to-end with figures
  embedded.
- `docs/` — decisions and their justification.
- `data/` — never holds data.

The forensic audit itself was performed in a separate workspace
(`C:\Projects\ML_Dataset_Audit\NHAMCS_2022`) and is not duplicated here.

## Reproducibility

1. `pip install -r requirements.txt`
2. Obtain `ed2022-stata.dta` from the official CDC/NCHS source (see
   [`data/README.md`](data/README.md)).
3. Point the code at it — no hard-coded absolute path. In order of precedence:
   - an explicit argument to `config.resolve_dataset_path(...)`
   - the `NHAMCS_DATA_PATH` environment variable
   - a documented sibling location under `ML_Dataset_Audit\`
4. Run the notebook top to bottom.

The loader refuses to proceed unless the file's size and SHA-256 match the
pinned values, so a wrong or substituted dataset fails loudly instead of
silently producing different numbers. `RANDOM_STATE = 42`. The dataset is
read-only, excluded by `.gitignore`, and never committed.

## Preprocessing design

A single `Pipeline` around a four-block `ColumnTransformer`
(`remainder="drop"`), in which **every learned parameter is fitted inside
`Pipeline.fit` on training data only** — imputation medians, category
vocabularies, rare-category thresholds, and scaling moments.

| Block | Steps |
|---|---|
| Numeric | median imputer + missingness indicator → standard scaler |
| Binary flags | passthrough, unscaled |
| Categorical, low cardinality | constant imputer → one-hot |
| Categorical, RFV | constant imputer → one-hot, `min_frequency=10`, `handle_unknown="infrequent_if_exist"` |

The notebook demonstrates the boundary three ways rather than asserting it:
`NotFittedError` before `fit`, different scaler means and category counts for a
train-only fit versus a full-data fit, and a finite holdout matrix that was never
fitted on.

Sentinel codes are resolved **per variable**. `998` is a Doppler/palpation method
code in `PULSE` and `BPDIAS` but a real 99.8 °F reading in `TEMPF`; `AMBTRANSFER
= -7` is structural, not missing; `RFV*D = 0` means "no complaint recorded", not
missing. Nothing is clipped, winsorised, or deleted by an IQR or z-score rule.

## Figures

All four are committed under `reports/figures/` and embedded in the notebook.

| Figure | Shows |
|---|---|
| `fig01_target_distribution.png` | Severe five-class imbalance motivating macro-F1 |
| `fig02_missingness.png` | Missingness by feature, and by target class |
| `fig03_feature_relationships.png` | Acuity by arrival mode and by reason for visit |
| `fig04_hospital_variation.png` | Hospital-to-hospital variation motivating grouped validation |

## Git

| Branch | Purpose |
|---|---|
| `main` | Stable. Reviewed work only. |
| `dev` | Active development. All work happens here. |

Raw data is excluded by `.gitignore` on both branches. **No commit, push, or
branch change has been made for this work**; it sits uncommitted on `dev`.

## Status

**Assignment 1, Part 1 — complete. Part 2 — not started.**

| # | Item | Status |
|---|---|---|
| 1 | Instructor dataset approval | **Outstanding** — stated precondition in the course brief |
| 2 | Target formulation and success metric | **Settled** — five-class `IMMEDR`, macro-F1 |
| 3 | Demographic-imputation policy | **Settled** — excluded as target-encoded leakage |
| 4 | NCHS confirmation on redistribution | **Outstanding** — mitigated by git-ignoring the raw file |
| 5 | Feature register | **Settled** — 18 raw → 20 engineered |
| 6 | Leakage-free preprocessing pipeline | **Implemented and verified** |
| 7 | Classifier, tuning, performance measurement | **Not started — Part 2** |
| 8 | Hospital-grouped validation | **Not started — Part 2** |

## Academic integrity

The course brief treats uncredited use of code from the internet or AI tools as
a violation. AI-assisted work in this repository is expected to be attributed in
the notebook.
