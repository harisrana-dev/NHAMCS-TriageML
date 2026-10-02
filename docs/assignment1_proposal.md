# Assignment 1, Part 1 — Proposal

**Course:** CS-13410 Machine Learning for Data Science
**Module:** Assignment 1, Part 1 — Problem framing and reproducible preprocessing
**Status:** implemented and verified against the official file; **no model has been trained**
**Branch:** `dev` (no commits made)

---

## 1. Problem

Triage in United States Emergency Service Areas (ESAs) assigns every arrival an
acuity level that determines how quickly the patient should be seen. The level is
assigned from the presenting complaint, a limited set of vital signs, and
demographic and administrative context.

The assignment is to build a reproducible, leakage-free machine-learning pipeline
over the **2022 NHAMCS Emergency Department Public Use File** that predicts triage
acuity. The prediction target is fixed by the brief and is **not** a choice made in
this proposal.

This document covers Part 1 only: framing, target definition, feature selection,
data quality, and the preprocessing design. Classifier selection, tuning, and
performance measurement belong to Part 2 and are deliberately absent here.

## 2. Dataset and target

| Property | Value |
|---|---|
| Source | NCHS / CDC, *National Hospital Ambulatory Medical Care Survey: 2022 Emergency Department Summary Tables* |
| File | `ed2022-stata.dta` |
| Rows / columns | 16,025 visits / 913 variables |
| Size / SHA-256 | 37,720,094 bytes / `0E00CD5AEA4511CD619F7AF1E0DC2F6825F111B7C87D3EBBB1358410A4C881C2` |
| Labelling | Triage level recorded by facility staff, not an outcome derived from later care |

**Target: `IMMEDR`, the nurse-recorded triage level, as a five-class problem.**

| Code | Level | n | % |
|---|---|---|---|
| 1 | Immediate | 139 | 1.36 |
| 2 | Emergent | 1,595 | 15.63 |
| 3 | Urgent | 5,340 | 52.32 |
| 4 | Semi-urgent | 2,826 | 27.69 |
| 5 | Non-urgent | 307 | 3.01 |

Modelling population: **10,207 visits (63.68% of the file)**, being those with a
valid `IMMEDR` in 1–5. The remaining 5,818 records are excluded for documented
reasons, not imputed: `-8` unknown (3,788), `7` visited an ESA that does not
conduct nursing triage (1,079), `0` no triage recorded although the ESA does
conduct triage (587), `-9` blank (364). Excluding `0` and `7` matters — they are
*documentation states*, not urgency levels, and folding them into the target would
invent a sixth and seventh class that no clinician chose.

The classes are **ordinal**: urgency decreases monotonically from 1 to 5. Two
consequences are recorded now and carried into Part 2: the primary metric is
macro-averaged so that the 1.36% Immediate class is not drowned out by the 52.32%
Urgent class, and a majority-class baseline of **0.5232** is the floor any model
must beat. These are baseline arithmetic, not achieved performance.

## 3. Research question

> **How accurately can five-class ED triage acuity (IMMEDR 1–5) be predicted at
> arrival from timing, vital signs, and documented reason-for-visit information
> in the 2022 NHAMCS ED dataset, and how much of that performance survives when
> evaluation holds out whole hospitals?**

The second clause is not a formality. The file covers **147 hospitals**, and
hospitals differ systematically in triage policy and documentation. A model
validated by random row splits will overestimate performance precisely because it
has seen each hospital's idiosyncrasies during training.

## 4. Features

**18 raw variables → 20 engineered columns.** Full detail, including sentinel
rules and level counts, is in [`feature_register.md`](feature_register.md); the
selection logic is in [`methodology.md`](methodology.md).

| Block | Columns | Raw variables |
|---|---|---|
| Numeric measurements | 7 | `ARRTIME` `TEMPF` `PULSE` `RESPR` `BPSYS` `BPDIAS` `POPCT` |
| Binary method flags | 2 | `PULSE` `BPDIAS` (Doppler / palpated provenance) |
| Low-cardinality categorical | 6 | `VMONTH` `VDAYR` `RESIDNCE` `ARREMS` `AMBTRANSFER` `SEEN72` |
| Reason-for-visit, 4-digit | 5 | `RFV13D` `RFV23D` `RFV33D` `RFV43D` `RFV53D` |

Deliberately excluded, on the evidence of the audit in
`C:\Projects\ML_Dataset_Audit\NHAMCS_2022\audit\`:

- **794 class-C variables** recorded *after* the triage decision, and **5
  class-D** variables derived from them.
- **All demographics** — `AGE`, `SEX`, `RACER`, `RACERETH`, `RACEUN`, `ETHIM`,
  `ETHUN`. These are not merely "unavailable before triage". NCHS imputed them
  using a model whose **inputs included triage level**, so they encode the target
  by construction. Using them would be leakage dressed as a feature.
- **41 class-F facility-only constants**, and all class-E keys and survey-design
  variables, including `HOSPCODE`.
- **`PAYOR`/`PAYOTHC` and `PAINSCALE`/`EPISODE`**, whose collection point relative
  to triage is *unresolved* in the official documentation. No assumption can be
  honestly stated for them, so they are dropped rather than guessed at.
- **The 5-digit RFV forms** and 24 class-G variables with unresolved timing.

**The one substantive assumption.** Reason-for-visit is retained only as the
4-digit codes, and only under a stated assumption:

> The presenting complaint is stated before the triage decision, and the released
> RFV code faithfully abstracts it.

The official documentation records counter-evidence to this: RTI assigned the
released code *retrospectively* from the verbatim complaint, with a documented
2.4% overall recoding error rate. The assumption is therefore disclosed, not
asserted, and the sensitivity of Part 2 results to dropping RFV is the first thing
to be checked there.

## 5. Data quality

Sentinel codes are resolved from the official codebook, and each is classified
rather than blanket-handled:

| Situation | Treatment |
|---|---|
| `-9` blank, `-8` unknown (categorical items) | Explicit categories, never imputed |
| `-7` not applicable (`AMBTRANSFER`) | Explicit structural category, never imputed |
| `0` in the RFV variables | Explicit structural category `none_recorded`, never imputed |
| `-9`/`-8` (numeric vitals) | `NaN` → median imputation **inside** the pipeline, with missingness indicators retained |
| `998` in `PULSE` / `BPDIAS` | Doppler / palpated estimate → value missing, **method flag retained** |
| `0` in `BPSYS` | Unobtainable → `NaN` |
| `998` in `TEMPF` | **A real value** (99.8 °F), not a method code. Retained. |

Nothing is clipped, winsorised, or dropped by an IQR or z-score rule. After
sentinel resolution, **every** observation falls inside its documented range, so
range-based deletion would discard real visits for no gain. Five `POPCT = 0`
values are physiologically impossible yet in-range; they are **retained and
reported**, because a value the codebook permits may still be a data-entry error,
and the decision of what to do with it belongs to Part 2, visibly, rather than
being silently absorbed into an imputer.

Result: after cleaning, **no categorical column contains a missing value**. All
missingness is confined to the seven numeric measurements (0.80%–8.27% each), and
is informative — the audit found it associated with acuity.

## 6. Preprocessing design

A single `sklearn` `Pipeline` containing a `ColumnTransformer` with four blocks.
**Every learned parameter is fitted inside `Pipeline.fit` on the training fold
only**: imputation medians, category vocabularies, rare-category thresholds, and
scaling moments. Nothing is precomputed on the full dataset. This is verified
mechanically in the notebook by showing that a full-data fit yields different
parameters than a train-only fit.

| Block | Steps |
|---|---|
| Numeric | median imputer (+ missing indicator) → standard scaler |
| Binary flags | passthrough, unscaled — a flag standardised into a z-score stops reading as a flag |
| Low-cardinality categorical | constant imputer → one-hot, `handle_unknown="ignore"` |
| Reason-for-visit (high cardinality) | constant imputer → one-hot, `min_frequency=10`, `handle_unknown="infrequent_if_exist"` |

`handle_unknown="infrequent_if_exist"` is required rather than cosmetic: grouped
validation holds out whole hospitals, so every evaluation fold contains complaint
categories absent from training. Mapping them into the rare-category bucket is
meaningful; plain `"ignore"` would emit an all-zero row that a linear model reads
as "nothing was observed" — indistinguishable from an unremarkable visit.

Grouped hospital-level validation, the second half of the research question, is
**implemented in Part 2**. Part 1 fixes only the technical boundary: no fitting
outside the pipeline, and `HOSPCODE` excluded as a feature while remaining
available as a grouping variable.

## 7. Responsible use and limitations

This is an academic exercise on de-identified public survey data. The output is
**not a clinical decision-support tool and must not be used to triage patients.**
Triage acuity reflects local protocol, staffing and clinician judgement; a model
trained on 2022 national survey data has no authority over an individual patient
in any specific emergency department.

Specific limitations carried forward:

1. **Class imbalance.** Immediate is 1.36%; macro-averaged F1 is the primary
   metric for this reason.
2. **Hospital clustering.** 147 hospitals with differing protocol. Random splits
   will overstate performance.
3. **RFV provenance.** Assigned retrospectively; 2.4% recoding error.
4. **Demographics excluded.** Including them would raise apparent accuracy while
   leaking the target through NCHS imputation, and would import documented racial
   disparities in imputation accuracy into the model.
5. **Missingness is informative but not causal.** A missing blood pressure is not
   a random event.
6. **No performance claim is made.** Part 1 reports no accuracy figure, because
   none has been measured.

## 8. Reproducibility

`src/config.py` resolves the dataset from an explicit argument, then
`NHAMCS_DATASET_PATH`, then a documented sibling location — never a hard-coded
absolute path. The loader refuses to proceed unless the file's size **and** SHA-256
match the pinned values, so a wrong or substituted file fails loudly. Random
state is fixed at 42. The dataset is read-only, is excluded by `.gitignore`, and
is not committed.

## 9. Open item

**Instructor approval of the dataset has not been recorded.** The audit lists this
as outstanding, and it gates submission regardless of the state of the code.
Everything else in Part 1 is implemented and verified.

## 10. Sources

- NCHS/CDC NHAMCS documentation: <https://www.cdc.gov/nchs/nhamcs/documentation/>
- Stata distribution: <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ed2022-stata.zip>
- Codebook `doc22-ed-508.pdf`, `readme2022-ed-stata.txt`, `eddict2022.dct`, `ed2022.do`
- Internal audit: `C:\Projects\ML_Dataset_Audit\NHAMCS_2022\audit\`
