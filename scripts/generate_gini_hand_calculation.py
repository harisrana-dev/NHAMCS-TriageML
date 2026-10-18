"""Issue #4: emit a hand-checkable Gini calculation from verified NHAMCS training rows.

Run at repository root: python scripts/generate_gini_hand_calculation.py
Requires the pinned ed2022-stata.dta, resolved by src.config.
"""
from pathlib import Path
from fractions import Fraction

from src.data.dataset import load_raw_dataset
from src.data.target import select_modelling_population
from src.evaluation.data import prepare_modelling_inputs
from src.evaluation.splits import make_grouped_folds
from src.preprocessing.features import FEATURES
from src import config

FOLD_ID = 0
FEATURE = "arrival_by_ambulance"
YES_CODE = "1"
LABELS = (1, 2, 3, 4, 5)
OUTPUT = Path("docs/gini_hand_calculation.md")


def counts_by_class(labels):
    return [int((labels == k).sum()) for k in LABELS]


def gini_exact(counts):
    n = sum(counts)
    if not n:
        raise ValueError("Empty child: chosen split cannot be evaluated")
    return Fraction(n * n - sum(c * c for c in counts), n * n)


def sum_str(counts):
    return " + ".join(str(c) for c in counts)


def squared_str(counts):
    return " + ".join(f"{c}^2" for c in counts)


def node_work(title, counts):
    n = sum(counts)
    g = gini_exact(counts)
    sq = sum(c*c for c in counts)
    return f"""### {title}

Class counts (IMMEDR 1, 2, 3, 4, 5): **{', '.join(map(str, counts))}**.

N = {sum_str(counts)} = **{n}**.

\\[
G_{{{title}}} = 1 - \\left[({counts[0]}/{n})^2 + ({counts[1]}/{n})^2 + ({counts[2]}/{n})^2 + ({counts[3]}/{n})^2 + ({counts[4]}/{n})^2\\right]
\\]

Squared-count sum = {squared_str(counts)} = **{sq}**.

\\[
G_{{{title}}} = 1 - \\frac{{{sq}}}{{{n*n}}} = \\frac{{{g.numerator}}}{{{g.denominator}}} = \\mathbf{{{float(g):.9f}}}
\\]
"""


def main():
    if FEATURE not in FEATURES:
        raise AssertionError(f"Feature {FEATURE!r} is not approved")
    raw = load_raw_dataset()  # built-in size, hash and shape verification
    pop = select_modelling_population(raw)
    inp = prepare_modelling_inputs(pop)
    folds = make_grouped_folds(inp.y, inp.groups,
                               n_splits=5, shuffle=True, random_state=42)
    fold = folds[FOLD_ID]
    x = inp.X.iloc[fold.train_positions][FEATURE].astype('string')
    y = inp.y.iloc[fold.train_positions]
    if not x.index.equals(y.index):
        raise AssertionError("X/y alignment lost")
    if x.isna().any():
        raise AssertionError("Unexpected null category; check preprocessing")
    left = (x == YES_CODE).fillna(False)
    parent = counts_by_class(y)
    ambulance = counts_by_class(y.loc[left])
    other = counts_by_class(y.loc[~left])
    n = sum(parent)
    a, b = sum(ambulance), sum(other)
    if not a or not b:
        raise AssertionError("Split yields empty child; do not produce a report")
    if [ambulance[i] + other[i] for i in range(5)] != parent:
        raise AssertionError("Counts do not reconcile")
    if len(fold.train_hospitals & fold.test_hospitals):
        raise AssertionError("Hospital leakage")
    gp, ga, gb = map(gini_exact, (parent, ambulance, other))
    weighted_a, weighted_b = Fraction(a, n)*ga, Fraction(b, n)*gb
    weighted = weighted_a + weighted_b
    gain = gp - weighted
    level_counts = x.value_counts(dropna=False).sort_index()
    levels = '\n'.join(f'| `{cat}` | {int(count)} | {"Yes child" if cat == YES_CODE else "Other child"} |'
                       for cat, count in level_counts.items())
    table = '\n'.join(f'| {k} | {parent[i]} | {ambulance[i]} | {other[i]} |'
                      for i, k in enumerate(LABELS))
    doc = f"""# Assignment 1 — Part 2: hand-worked Gini split (Issue #4)

**Result type:** Manually evaluated **candidate** split on actual training observations; **not** the split chosen by a trained Decision Tree. No tree was fitted for this calculation.

## Source and data provenance

- Raw source: **2022 NHAMCS Emergency Department**, official `ed2022-stata.dta`.
- Raw file SHA-256 (verified by repository loader): `{config.EXPECTED_SHA256}`.
- Raw expected shape: {config.EXPECTED_SHAPE[0]:,} rows × {config.EXPECTED_SHAPE[1]} columns.
- Modelling-population rows (IMMEDR 1–5): **{len(pop):,}**.
- Existing functions used unchanged: `src.data.dataset.load_raw_dataset`, `src.data.target.select_modelling_population`, `src.evaluation.data.prepare_modelling_inputs`, `src.evaluation.splits.make_grouped_folds`.
- Split: `StratifiedGroupKFold`, `n_splits=5`, `shuffle=True`, `random_state=42`, grouped by `HOSPCODE`.
- Chosen fold: **fold_id={FOLD_ID}**, zero-based as emitted by `make_grouped_folds`.
- **Only the training positions of that fold** enter the arithmetic: **{n} records**; held-out test rows: **{len(fold.test_positions)}**, not examined for candidate selection or calculations.
- Distinct training hospitals: **{len(fold.train_hospitals)}**; distinct test hospitals: **{len(fold.test_hospitals)}**; intersection: **0**.
- Subset rule: **all training rows from fold {FOLD_ID}**, no subsampling, no selection based on target labels or impurity values.

The course brief requires a hand-worked information-gain **or** Gini split; using actual project training data, grouped-fold training rows and a predetermined selection rule are stricter **project implementation requirements**, not quotations from the course brief.

## Candidate predictor and predetermined split

- Approved engineered predictor: **`{FEATURE}`**, sourced from raw `ARREMS`.
- Rule: **left/Yes child** if the engineered category is the exact string **`{YES_CODE}`**; **right/Other child** for *every other value*.
- `blank` (raw −9), `unknown` (raw −8), and any `missing` state are explicitly assigned to **Other** rather than dropped. All observed feature categories are enumerated below.
- The rule was specified before computing class counts. It is a binary grouping solely for manual illustration; the model preprocessing configuration is unchanged.
- `HOSPCODE` is used **only** as a grouping variable and is never a predictor. Identifiers, target-imputed demographics, and disallowed features are not used.

### Training-fold category audit

| Engineered `arrival_by_ambulance` value | N | Destination |
|---|---:|---|
{levels}

**Semantic check:** confirm against the official `ARREMS` codebook/value labels that code `1` means ambulance arrival before describing the Yes child clinically. The arithmetic and literal-code split do not depend on that interpretation.

## Observed training counts

| IMMEDR class | Parent | Child A (code `1`) | Child B (all other codes) |
|---|---:|---:|---:|
{table}
| **Total** | **{n}** | **{a}** | **{b}** |

Every row and total reconcile: parent counts are exactly the sums of corresponding children. `IMMEDR` values other than 1–5 were excluded by the project's existing population rule, not by this calculation.

## Hand-worked arithmetic

For a node with class counts \\(c_1, c_2, c_3, c_4, c_5\\) and size \\(N = \\sum_i c_i\\):

\\[G = 1 - \\sum_{{i=1}}^5 (c_i/N)^2\\]

{node_work('Parent', parent)}
{node_work('A', ambulance)}
{node_work('B', other)}

### Weighted child impurity

\\[
G_{{weighted}} = (N_A/N)G_A + (N_B/N)G_B
\\]

\\[
G_{{weighted}} = ({a}/{n})({float(ga):.9f}) + ({b}/{n})({float(gb):.9f})
\\]

Individual weighted terms: \\(\\frac{{{weighted_a.numerator}}}{{{weighted_a.denominator}}} = {float(weighted_a):.9f}\\) and \\(\\frac{{{weighted_b.numerator}}}{{{weighted_b.denominator}}} = {float(weighted_b):.9f}\\).

\\[
G_{{weighted}} = \\frac{{{weighted.numerator}}}{{{weighted.denominator}}} = \\mathbf{{{float(weighted):.9f}}}
\\]

### Gini reduction

\\[
\\Delta G = G_{{Parent}} - G_{{weighted}}
\\]

\\[
\\Delta G = {float(gp):.9f} - {float(weighted):.9f} = \\frac{{{gain.numerator}}}{{{gain.denominator}}} = \\boxed{{{float(gain):.9f}}}
\\]

## Interpretation and limitations

The positive reduction, if observed, indicates that the stated **candidate split** yields children with lower average impurity than their parent. A value near zero would indicate little improvement. The sign and magnitude above are computed from real training-fold counts, not a fitted-tree split or test-fold optimization.

The split is not claimed to be optimal, nor selected by a fitted classifier. No predictive accuracy or cross-validated score can be inferred from this single candidate Gini calculation. The complete fold training set was used for transparent reproducibility; the illustrative binary recoding is **not** a change to the learning pipeline.

## Reproduction

From the repository root, with the verified dataset discoverable by `src.config`:

```bash
python scripts/generate_gini_hand_calculation.py
```

This regenerates `docs/gini_hand_calculation.md` by resolving and verifying the dataset, invoking the project population/feature/fold functions, counting classes, and calculating **exact rational** Gini values (formatted to nine decimal places). No model is fitted and no test-fold rows are used in arithmetic.
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(doc, encoding='utf-8')
    print(f"Wrote {OUTPUT}; fold {FOLD_ID}, N={n}, Gini reduction={float(gain):.9f}")


if __name__ == "__main__":
    main()
