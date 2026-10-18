# Assignment 1 — Part 2: hand-worked Gini split (Issue #4)

**Result type:** Manually evaluated **candidate** split on actual training observations; **not** the split chosen by a trained Decision Tree. No tree was fitted for this calculation.

## Source and data provenance

- Raw source: **2022 NHAMCS Emergency Department**, official `ed2022-stata.dta`.
- Raw file SHA-256 (verified by repository loader): `0E00CD5AEA4511CD619F7AF1E0DC2F6825F111B7C87D3EBBB1358410A4C881C2`.
- Raw expected shape: 16,025 rows × 913 columns.
- Modelling-population rows (IMMEDR 1–5): **10,207**.
- Existing functions used unchanged: `src.data.dataset.load_raw_dataset`, `src.data.target.select_modelling_population`, `src.evaluation.data.prepare_modelling_inputs`, `src.evaluation.splits.make_grouped_folds`.
- Split: `StratifiedGroupKFold`, `n_splits=5`, `shuffle=True`, `random_state=42`, grouped by `HOSPCODE`.
- Chosen fold: **fold_id=0**, zero-based as emitted by `make_grouped_folds`.
- **Only the training positions of that fold** enter the arithmetic: **8167 records**; held-out test rows: **2040**, not examined for candidate selection or calculations.
- Distinct training hospitals: **116**; distinct test hospitals: **31**; intersection: **0**.
- Subset rule: **all training rows from fold 0**, no subsampling, no selection based on target labels or impurity values.

The course brief requires a hand-worked information-gain **or** Gini split; using actual project training data, grouped-fold training rows and a predetermined selection rule are stricter **project implementation requirements**, not quotations from the course brief.

## Candidate predictor and predetermined split

- Approved engineered predictor: **`arrival_by_ambulance`**, sourced from raw `ARREMS`.
- Rule: **left/Yes child** if the engineered category is the exact string **`1`**; **right/Other child** for *every other value*.
- `blank` (raw −9), `unknown` (raw −8), and any `missing` state are explicitly assigned to **Other** rather than dropped. All observed feature categories are enumerated below.
- The rule was specified before computing class counts. It is a binary grouping solely for manual illustration; the model preprocessing configuration is unchanged.
- `HOSPCODE` is used **only** as a grouping variable and is never a predictor. Identifiers, target-imputed demographics, and disallowed features are not used.

### Training-fold category audit

| Engineered `arrival_by_ambulance` value | N | Destination |
|---|---:|---|
| `1` | 1515 | Yes child |
| `2` | 6539 | Other child |
| `blank` | 42 | Other child |
| `unknown` | 71 | Other child |

**Semantic check:** confirm against the official `ARREMS` codebook/value labels that code `1` means ambulance arrival before describing the Yes child clinically. The arithmetic and literal-code split do not depend on that interpretation.

## Observed training counts

| IMMEDR class | Parent | Child A (code `1`) | Child B (all other codes) |
|---|---:|---:|---:|
| 1 | 114 | 67 | 47 |
| 2 | 1276 | 419 | 857 |
| 3 | 4259 | 829 | 3430 |
| 4 | 2272 | 173 | 2099 |
| 5 | 246 | 27 | 219 |
| **Total** | **8167** | **1515** | **6652** |

Every row and total reconcile: parent counts are exactly the sums of corresponding children. `IMMEDR` values other than 1–5 were excluded by the project's existing population rule, not by this calculation.

## Hand-worked arithmetic

For a node with class counts \(c_1, c_2, c_3, c_4, c_5\) and size \(N = \sum_i c_i\):

\[G = 1 - \sum_{i=1}^5 (c_i/N)^2\]

### Parent

Class counts (IMMEDR 1, 2, 3, 4, 5): **114, 1276, 4259, 2272, 246**.

N = 114 + 1276 + 4259 + 2272 + 246 = **8167**.

\[
G_{Parent} = 1 - \left[(114/8167)^2 + (1276/8167)^2 + (4259/8167)^2 + (2272/8167)^2 + (246/8167)^2\right]
\]

Squared-count sum = 114^2 + 1276^2 + 4259^2 + 2272^2 + 246^2 = **25002753**.

\[
G_{Parent} = 1 - \frac{25002753}{66699889} = \frac{41697136}{66699889} = \mathbf{0.625145508}
\]

### A

Class counts (IMMEDR 1, 2, 3, 4, 5): **67, 419, 829, 173, 27**.

N = 67 + 419 + 829 + 173 + 27 = **1515**.

\[
G_{A} = 1 - \left[(67/1515)^2 + (419/1515)^2 + (829/1515)^2 + (173/1515)^2 + (27/1515)^2\right]
\]

Squared-count sum = 67^2 + 419^2 + 829^2 + 173^2 + 27^2 = **897949**.

\[
G_{A} = 1 - \frac{897949}{2295225} = \frac{1397276}{2295225} = \mathbf{0.608775175}
\]

### B

Class counts (IMMEDR 1, 2, 3, 4, 5): **47, 857, 3430, 2099, 219**.

N = 47 + 857 + 3430 + 2099 + 219 = **6652**.

\[
G_{B} = 1 - \left[(47/6652)^2 + (857/6652)^2 + (3430/6652)^2 + (2099/6652)^2 + (219/6652)^2\right]
\]

Squared-count sum = 47^2 + 857^2 + 3430^2 + 2099^2 + 219^2 = **16955320**.

\[
G_{B} = 1 - \frac{16955320}{44249104} = \frac{3411723}{5531138} = \mathbf{0.616821168}
\]


### Weighted child impurity

\[
G_{weighted} = (N_A/N)G_A + (N_B/N)G_B
\]

\[
G_{weighted} = (1515/8167)(0.608775175) + (6652/8167)(0.616821168)
\]

Individual weighted terms: \(\frac{1397276}{12373005} = 0.112929398\) and \(\frac{6823446}{13581721} = 0.502399217\).

\[
G_{weighted} = \frac{12661190678}{20576307315} = \mathbf{0.615328615}
\]

### Gini reduction

\[
\Delta G = G_{Parent} - G_{weighted}
\]

\[
\Delta G = 0.625145508 - 0.615328615 = \frac{1649696542294}{168046701841605} = \boxed{0.009816893}
\]

## Interpretation and limitations

The positive reduction, if observed, indicates that the stated **candidate split** yields children with lower average impurity than their parent. A value near zero would indicate little improvement. The sign and magnitude above are computed from real training-fold counts, not a fitted-tree split or test-fold optimization.

The split is not claimed to be optimal, nor selected by a fitted classifier. No predictive accuracy or cross-validated score can be inferred from this single candidate Gini calculation. The complete fold training set was used for transparent reproducibility; the illustrative binary recoding is **not** a change to the learning pipeline.

## Reproduction

From the repository root, with the verified dataset discoverable by `src.config`:

```bash
python scripts/generate_gini_hand_calculation.py
```

This regenerates `docs/gini_hand_calculation.md` by resolving and verifying the dataset, invoking the project population/feature/fold functions, counting classes, and calculating **exact rational** Gini values (formatted to nine decimal places). No model is fitted and no test-fold rows are used in arithmetic.
