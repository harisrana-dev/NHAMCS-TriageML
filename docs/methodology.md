# Methodology — Assignment 1, Part 1

Scope: target definition, feature selection, data quality, and the preprocessing
design. **No classifier is trained and no performance figure is reported**, both
belonging to Part 2.

## 1. Target formulation

`IMMEDR`, the nurse-recorded triage level, is used as a **five-class** problem
(1–5). The alternative formulations considered in
`audit/target_binary_formulations.csv` were rejected: collapsing 1–2 vs 3–5
discards the immediate-versus-emergent distinction that has the largest clinical
consequence, and a binarisation at 3 vs 1–2/4–5 is uninterpretable given class
frequencies of 1.36% / 52.32% / 27.69%.

Modelling population = 10,207 of 16,025 visits (63.68%).

| Excluded `IMMEDR` | n | % of file | Why excluded |
|---|---|---|---|
| `-8` Unknown | 3,788 | 23.64 | No label exists. |
| `7` ESA does not conduct nursing triage | 1,079 | 6.73 | Different population; no triage to predict. |
| `0` No triage recorded, ESA does conduct triage | 587 | 3.66 | Documentation state, not an urgency level. |
| `-9` Blank | 364 | 2.27 | No label exists. |

`0` and `7` are the consequential exclusions. Both are *recording states*, and
treating them as classes would ask the model to predict whether a facility
documented its triage decision — a different question from clinical urgency.

**Primary metric: macro-averaged F1**, fixed by the brief. Justified independently:
the classes are ordinal and severely imbalanced, so macro-averaging prevents the
52.32% Urgent class from dominating the score while the 1.36% Immediate class is
ignored. **Majority-class baseline: 0.5232** — the floor to beat.

## 2. Feature selection

Selection was driven by an external forensic audit of all 913 variables
(`C:\Projects\ML_Dataset_Audit\NHAMCS_2022\audit\`), not by inspection of
correlations with the target. The governing question for each variable is *when
was it recorded relative to the triage decision*; a variable that cannot be shown
to be pre-decision is excluded.

Result: **18 raw variables → 20 engineered columns**. Full table in
[`feature_register.md`](feature_register.md).

The three exclusions that carry the most methodological weight:

1. **Demographics are leakage.** `AGE`, `SEX`, `RACER`, `RACERETH`, `RACEUN`,
   `ETHIM`, `ETHUN` were imputed by NCHS using a model that took triage level as
   an input. They are class C, not class E. Using them would produce a target
   shortcut with no clinical content.
2. **Unresolved timing is excluded, not assumed away.** `PAYOR`, `PAINSCALE`,
   `EPISODE` and 24 class-G variables are dropped because the documentation is
   silent on their collection point. The RFV variables are retained *only*
   because a specific assumption can be stated for them, and that assumption is
   disclosed with its counter-evidence. Because RFV reaches Cramer's V 0.41–0.51
   against the target — the strongest signal in the file — the audit calls this
   "the single most consequential judgement in the audit". If the assumption is
   wrong, the measured performance is wrong with it.
3. **`HOSPCODE` is excluded as a feature but retained as a grouping variable.**
   It must not enter the matrix, and it defines the grouping structure Part 2
   needs. `src/preprocessing/features.py::leakage_checks()` enforces its absence
   and is called at pipeline construction, so a later edit cannot reintroduce it.

## 3. Data quality

### Principle: classify each sentinel, never handle them uniformly

A blanket rule such as "drop all negatives" or "null out everything ≤ 0" would be
wrong here, because these codes mean three genuinely different things:

- **Recorded response states** (`−9` blank, `−8` unknown) — the item was asked
  and not answered. Informative; explicit categories, never imputed.
- **Structural non-applicability** (`AMBTRANSFER = −7`; `RFV*D = 0`) — the item
  does not exist for this visit. **Not missing data.** `AMBTRANSFER = −7` covers
  8,226 of 10,207 visits (80.6%), because the item only applies to ambulance
  arrivals; reporting it as missing would invent a 80% missingness rate and then
  impute away a genuinely meaningful structural fact.
- **Method codes and unobtainable values** (`998`, `BPSYS = 0`) — a measurement
  was attempted and obtained by a different route, or not obtained.

### The 998 trap

`998` is a method code in `PULSE` and `BPDIAS` but **a real value in `TEMPF`**.
Treated uniformly, `TEMPF = 998` would discard 26 perfectly good 99.8 °F readings;
treated as a value everywhere, `PULSE = 998` would inject an impossible 998 bpm.
The distinction is per-variable, taken from the codebook.

For the genuine method codes, the value becomes `NaN` **and a binary flag is
retained**, so the model can tell "not measured" from "measured by Doppler /
palpation". Those flags are deliberately outside the scaled numeric block: a 0/1
flag standardised into a z-score stops reading as a flag.

Note `bp_diastolic_palpated_or_doppler` has only **1** positive record in the
modelling population, so it is constant in most training folds. It is kept for
correctness and disclosed as effectively uninformative at this sample size.

### No distribution-based deletion

No IQR rule, no z-score rule, no winsorisation, no clipping. After sentinel
resolution every observation lies inside its documented range, so such rules
would delete real visits for no analytic gain. The one clearly erroneous residual
— 5 × `POPCT = 0` — is reported and retained.

### Missingness

After cleaning, **no categorical column contains any missing value**; all
missingness sits in the 7 numeric measurements, between 0.80% (`arrival_minutes`)
and 8.27% (`bp_diastolic`). Missingness is informative rather than random — the
audit found it associated with acuity (Cramer's V 0.066–0.137) — so it is
preserved explicitly via `SimpleImputer(add_indicator=True)` rather than being
absorbed into the imputed value.

## 4. Preprocessing

A single `Pipeline` around a four-block `ColumnTransformer`, with
`remainder="drop"`.

| Block | Steps | Notes |
|---|---|---|
| Numeric (7) | median imputer + missing indicator → `StandardScaler` | medians and moments learned in `fit` |
| Binary flags (2) | `passthrough` | unscaled; never imputed |
| Categorical, low cardinality (6) | constant imputer → one-hot, `handle_unknown="ignore"` | vocabularies learned in `fit` |
| Categorical, RFV (5) | constant imputer → one-hot, `min_frequency=10`, `handle_unknown="infrequent_if_exist"` | threshold learned in `fit` |

### The leakage boundary

**Every learned parameter is fitted inside `Pipeline.fit` on training data only.**
No imputation, encoding, grouping, or scaling is computed outside the pipeline.
The notebook verifies this mechanically rather than by assertion: it fits on the
training fold, then shows that a full-data fit produces *different* parameters
(scaler means differ; 317 RFV categories vs 326).

### Why `infrequent_if_exist`

Grouped validation holds out whole hospitals, so every evaluation fold contains
complaint categories never seen in training. `"ignore"` maps such a category to an
all-zero row, which a linear model reads as "nothing was observed" — the same
encoding as an unremarkable visit. `"infrequent_if_exist"` maps both unseen and
rare categories into the single infrequent column, which is the honest encoding.
On a 7,655-row training fold this yields 439 RFV columns with **zero** dead
all-zero columns.

### Row-local cleaning

All sentinel handling is row-local: no value depends on any other row, and the
output is invariant to row order. Verified in the notebook by cleaning a shuffled
copy and asserting equality. This matters because it means the cleaning step
cannot itself be a vector of information transfer.

## 5. Validation strategy

**Part 1 fixes only the technical boundary.** Implemented and verified here:

- the pipeline is not fitted before `fit` is called (`NotFittedError` check);
- parameters are learned on the training fold only, demonstrated by contrast with
  a full-data fit;
- train and evaluation folds transform to identical column counts;
- the transformed evaluation matrix contains no `NaN` and no `inf`;
- no category is a frequency statistic computed on the full dataset.

**Deferred to Part 2:** hospital-grouped validation. The plan is repeated
stratified group k-fold cross-validation with `HOSPCODE` as the grouping
variable, reporting per-hospital generalization, not just pooled. `HOSPCODE`
covers **147 hospitals** and is constant within facility, which is why the 41
class-F facility-only variables are excluded as features.

## 6. Reproducibility

- `src/config.py` resolves the dataset from an explicit argument, then
  `NHAMCS_DATASET_PATH`, then a documented sibling location. No hard-coded
  absolute path; no network access.
- The loader verifies **size and SHA-256** against pinned values and refuses to
  proceed on mismatch, so a wrong or substituted file fails loudly.
- `convert_categoricals=False` is mandatory: pandas' default converter raises on
  this official file because the `TEMPF` value-label set contains a duplicate.
- `RANDOM_STATE = 42`.
- The dataset is read-only, excluded by `.gitignore`, and not committed.
- **No commit, push, or branch change has been made.** Work is on `dev`.

## 7. Known limitations

1. **Class imbalance** — Immediate 1.36%. Macro-F1 is the reason it is not ignored.
2. **Hospital clustering** — 147 hospitals with differing protocol; random splits
   overstate performance. The reason grouped validation is deferred, not skipped.
3. **RFV provenance** — assigned retrospectively from the verbatim complaint,
   with a documented 2.4% recoding error rate.
4. **Demographics excluded** — reduces apparent accuracy; including them would be
   leakage and would import imputation disparities.
5. **`POPCT = 0`** retained as implausible; treatment deferred to Part 2.
6. **`bp_diastolic_palpated_or_doppler`** has 1 positive record — effectively
   uninformative.
7. **No performance claim.** No model has been trained.
8. **Instructor dataset approval is unrecorded** and gates submission.

## 8. Sources

- <https://www.cdc.gov/nchs/nhamcs/documentation/>
- <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ed2022-stata.zip>
- Codebook: `doc22-ed-508.pdf`, `readme2022-ed-stata.txt`, `eddict2022.dct`, `ed2022.do`
- Audit: `C:\Projects\ML_Dataset_Audit\NHAMCS_2022\audit\`
