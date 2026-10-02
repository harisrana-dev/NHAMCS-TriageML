# Final Feature Register — Assignment 1, Part 1

Target: `IMMEDR` (five-class, 1–5). Population: 10,207 visits.
**18 raw source variables → 20 engineered columns.**

Every number in this register is computed from the live code
(`src/preprocessing/sentinels.py`, `src/preprocessing/features.py`) against the
official file, not transcribed by hand. Missing = genuinely absent after sentinel
resolution; recorded response states and structural non-applicability are counted
as *categories*, not missingness.

## Register

| Engineered column | Raw variable | Kind | Levels | Missing | Observed range | Sentinel codes |
|---|---|---|---|---|---|---|
| `arrival_minutes` | `ARRTIME` | numeric | – | 82 | 0 to 1439 | −9, −8, −7 |
| `temp_f` | `TEMPF` | numeric | – | 299 | 91.2 to 105.0 | −9, −8 |
| `pulse_bpm` | `PULSE` | numeric | – | 206 | 3 to 217 | −9, −8, 998 |
| `resp_rate` | `RESPR` | numeric | – | 239 | 1 to 150 | −9, −8 |
| `bp_systolic` | `BPSYS` | numeric | – | 828 | 43 to 289 | −9, −8, 0 |
| `bp_diastolic` | `BPDIAS` | numeric | – | 844 | 22 to 190 | −9, −8, 998 |
| `spo2_pct` | `POPCT` | numeric | – | 219 | 0 to 100 | −9, −8 |
| `pulse_measured_by_doppler` | `PULSE` | binary flag | 2 (38 set) | 0 | 0 to 1 | 998 |
| `bp_diastolic_palpated_or_doppler` | `BPDIAS` | binary flag | 2 (1 set) | 0 | 0 to 1 | 998 |
| `visit_month` | `VMONTH` | categorical (low) | 12 | 0 | – | none |
| `visit_day_of_week` | `VDAYR` | categorical (low) | 7 | 0 | – | none |
| `patient_residence` | `RESIDNCE` | categorical (low) | 6 | 0 | – | −9, −8 |
| `arrival_by_ambulance` | `ARREMS` | categorical (low) | 4 | 0 | – | −9, −8 |
| `interfacility_transfer` | `AMBTRANSFER` | categorical (low) | 5 | 0 | – | −9, −8, −7 *(structural)* |
| `seen_in_this_ed_within_72h` | `SEEN72` | categorical (low) | 4 | 0 | – | −9, −8 |
| `rfv1_broad_category` | `RFV13D` | categorical (RFV) | 326 | 0 | – | 0 *(structural)* |
| `rfv2_broad_category` | `RFV23D` | categorical (RFV) | 343 | 0 | – | 0 *(structural)* |
| `rfv3_broad_category` | `RFV33D` | categorical (RFV) | 326 | 0 | – | 0 *(structural)* |
| `rfv4_broad_category` | `RFV43D` | categorical (RFV) | 289 | 0 | – | 0 *(structural)* |
| `rfv5_broad_category` | `RFV53D` | categorical (RFV) | 256 | 0 | – | 0 *(structural)* |

`PULSE` and `BPDIAS` each contribute a value column *and* a method-flag column,
which is why 18 raw variables yield 20 engineered columns.

## Transformations applied

| Transformation | Rule |
|---|---|
| `ARRTIME` → minutes | `HH*60 + MM`; `−9`/`−8`/`−7` → `NaN`. Midnight `0000` → `0` and **retained** (it is a legitimate time, not a sentinel). |
| `TEMPF` → °F | ÷ 10. `−9`/`−8` → `NaN`. `998` → 99.8 °F, **retained** — it is a real value, not a method code. |
| `PULSE` | `−9`/`−8` → `NaN`; `998` → `NaN` **and** `pulse_measured_by_doppler = 1`. |
| `RESPR` | `−9`/`−8` → `NaN`. |
| `BPSYS` | `−9`/`−8` → `NaN`; `0` → `NaN` (unobtainable). |
| `BPDIAS` | `−9`/`−8` → `NaN`; `998` → `NaN` **and** `bp_diastolic_palpated_or_doppler = 1`. |
| `POPCT` | `−9`/`−8` → `NaN`. `0` retained. |
| `VMONTH`, `VDAYR` | Int64 → string category. |
| `RESIDNCE`, `ARREMS`, `SEEN72` | `−9` → `blank`, `−8` → `unknown`, as explicit categories. |
| `AMBTRANSFER` | `−7` → `not_applicable` (structural), `−9` → `blank`, `−8` → `unknown`. |
| `RFV*D` | `0` → `none_recorded` (structural), as an explicit category. |

All cleaning is **row-local**: no value is computed from any other row, no global
statistic is applied, and the result is invariant to row order. This is verified
in the notebook.

## Range validation — reported, never enforced

After sentinel resolution, **every** observation lies inside its documented range:

| Feature | Documented | Observed min | Observed median | Observed max | Out of range |
|---|---|---|---|---|---|
| `arrival_minutes` | 0–1439 | 0 | 838 | 1439 | 0 |
| `temp_f` | 89.6–105.6 | 91.2 | 98.1 | 105.0 | 0 |
| `pulse_bpm` | 0–240 | 3 | 89 | 217 | 0 |
| `resp_rate` | 0–150 | 1 | 18 | 150 | 0 |
| `bp_systolic` | 43–289 | 43 | 133 | 289 | 0 |
| `bp_diastolic` | 22–190 | 22 | 79 | 190 | 0 |
| `spo2_pct` | 0–100 | 0 | 98 | 100 | 0 |

Nothing is clipped, winsorised, or removed by an IQR or z-score rule.

**Flagged but retained:** 5 × `POPCT = 0`. In-range, yet not a physiologically
survivable saturation, and therefore most likely data-entry error. Retained and
reported rather than silently imputed or deleted, so the choice is visible to
Part 2.

## Exclusions, with reasons

| Class | n | Decision | Reason |
|---|---|---|---|
| A — timing-eligible | 13 | **Included** | Documented as collected before the triage decision. |
| B — RFV, 4-digit | 5 | **Included** | Assumption stated and disclosed below. |
| B — RFV, 5-digit | 5 | Excluded | The 5th digit encodes symptom-character detail not necessarily elicited before triage. |
| B — `PAYOR`/`PAYOTHC`, `PAINSCALE`, `EPISODE` | 12 | Excluded | Collection point relative to triage is **unresolved** in the documentation; no honest assumption available. |
| C — post-decision | 794 | Excluded | Recorded after the triage decision. |
| D — derived from C | 5 | Excluded | Inherits class-C timing. |
| E — keys/admin/design | 13 | Excluded | Includes `HOSPCODE`, `PATCODE`, `CPSUM`, `CSTRATM`. |
| F — facility-only | 41 | Excluded | Constant within `HOSPCODE`; a facility fingerprint, not a patient feature. |
| G — timing unresolved | 24 | Excluded | Documentation silent. |
| Demographics | 7 | Excluded | **NCHS imputed them using a model that took triage level as an input** — target-encoded by construction. |
| T — target | 1 | `IMMEDR` | The label. |

### Demographics are leakage, not merely "post-triage"

`AGE`, `SEX`, `RACER`, `RACERETH`, `RACEUN`, `ETHIM`, `ETHUN` are classed **C**
by the audit. NCHS imputed missing demographics using a model whose inputs
included the triage level, so the published value partially encodes the target.
Including them would inflate measured performance while teaching the model a
shortcut that has no clinical meaning — and would import documented disparities in
imputation accuracy by race into the predictions.

### Disclosed assumption — reason-for-visit

> The presenting complaint is stated before the triage decision, and the released
> RFV code faithfully abstracts it.

Counter-evidence, from the official documentation: RTI assigned the released code
*retrospectively* from the verbatim complaint, with a documented **2.4% overall
recoding error rate**. The code — not the complaint — is what enters the model.
This assumption is a disclosed risk, not an established fact, and Part 2 must
test sensitivity to dropping RFV entirely.

Its weight is easy to understate: RFV reaches **Cramer's V 0.41–0.51** against the
target, the strongest signal among all 913 variables. The audit therefore calls
this "the single most consequential judgement in the audit". If the assumption is
wrong, the measured performance is wrong with it.

## Within-pipeline parameters (fitted on training folds only)

| Parameter | Value | Where fitted |
|---|---|---|
| Numeric imputation | median | `SimpleImputer`, inside `Pipeline.fit` |
| Numeric missingness indicator | retained (`add_indicator=True`) | inside `fit` |
| Numeric scaling | standard | `StandardScaler`, inside `fit` |
| Binary flags | passthrough, unscaled | — (no learned parameter) |
| Low-cardinality one-hot | `handle_unknown="ignore"` | vocabulary learned inside `fit` |
| RFV one-hot | `min_frequency=10`, `handle_unknown="infrequent_if_exist"` | threshold and vocabulary learned inside `fit` |

`min_frequency=10` groups rare complaint categories into one shared bucket per
slot rather than dropping them, because rare categories are disproportionately
high-acuity. On a 7,655-row training fold this groups 193/208/221/205/186
categories across the five slots into 5 infrequent columns, with **zero** all-zero
dead columns.
