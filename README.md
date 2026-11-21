# NHAMCS-TriageML

Machine-learning study of emergency-department triage urgency using the 2022
NHAMCS Emergency Department public-use dataset.

> **Status: Part 1 complete; Part 2 models implemented, evaluated and
> integrated. Parts 3–5 not started.**
>
> Part 1 loads and verifies the official file byte-for-byte, settles the target
> and feature register, and demonstrates the leakage-free preprocessing pipeline.
> Part 2 trains three models (Dummy, GaussianNB, Gini Decision Tree) on one
> shared set of five-fold hospital-grouped splits under the fixed protocol below,
> and integrates the results into
> `notebooks/01_assignment1_part1_eda_and_pipeline.ipynb` (sections P2.1–P2.12)
> together with the Macro-F1 comparison figure. **Random Forest, AdaBoost,
> XGBoost, Logistic Regression, nested hyperparameter tuning, the MLP, PCA and
> K-means are not implemented — they are Part 3/4/5 work.** Instructor approval
> of the dataset is still outstanding.

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
├── README.md                    # this file
├── requirements.txt             # core dependencies (the one authoritative manifest)
├── requirements-xgboost.txt     # optional: XGBoost (Issue #13)
├── requirements-torch-cpu.txt   # optional: PyTorch CPU-only wheels (Issue #16)
├── data/README.md               # how to obtain and verify the dataset — no data files
├── notebooks/
│   └── 01_assignment1_part1_eda_and_pipeline.ipynb   # executed; Part 1 + Part 2, 5 figures inline
├── src/
│   ├── config.py          # dataset resolution, pinned identity, seed
│   ├── data/              # integrity verification + target definition
│   ├── preprocessing/     # sentinels, feature register, ColumnTransformer
│   └── evaluation/        # hospital-grouped evaluation framework (Part 2)
├── scripts/               # reproducible experiment entry points (Part 2)
├── docs/
│   ├── environment.md            # authoritative environment spec (Issue #23)
│   ├── assignment1_proposal.md   # the one-page proposal
│   ├── feature_register.md       # final 18 raw -> 20 engineered
│   ├── methodology.md            # why each decision was made
│   ├── gaussian_nb_experiment.md # Part 2: GaussianNB results (Issue #2)
│   ├── decision_tree_results.md  # Part 2: Decision Tree results (Issue #3)
│   └── gini_hand_calculation.md  # Part 2: hand-worked Gini split (Issue #4)
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

The authoritative environment guide is
[`docs/environment.md`](docs/environment.md) — supported Python, core and
optional installs, dataset path precedence, verification commands.

1. Create a Python 3.13 virtual environment and install the core manifest:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Model work that needs XGBoost or PyTorch adds
   `requirements-xgboost.txt` / `requirements-torch-cpu.txt` on top (see
   [`docs/environment.md`](docs/environment.md)).
2. Obtain `ed2022-stata.dta` from the official CDC/NCHS source (see
   [`data/README.md`](data/README.md)).
3. Point the code at it — no hard-coded absolute path. In order of precedence:
   - an explicit argument to `config.resolve_dataset_path(...)`
   - the `NHAMCS_DATA_PATH` environment variable
   - `data/local/ed2022-stata.dta` (the git-ignored drop location — the usual
     place to put the file)
   - a documented sibling location under `ML_Dataset_Audit\`
4. Run the notebook top to bottom. It now contains Part 1 (sections 1–8) and
   the Part 2 comparison (sections P2.1–P2.12).

The loader refuses to proceed unless the file's size and SHA-256 match the
pinned values, so a wrong or substituted dataset fails loudly instead of
silently producing different numbers. `RANDOM_STATE = 42`. The dataset is
read-only, excluded by `.gitignore`, and never committed.

**Reproducing the Part 2 experiment** (from the repository root, with the
dataset discoverable as above):

```bash
python -m pytest -q                  # full suite: 64 unit + 5 integration tests
python -m pytest -m integration -q   # 5 real-dataset integration tests
                                     # (without the dataset they skip; 64 unit still run)
python -m scripts.run_decision_tree  # single-model run through the shared evaluator
python -m scripts.fold_metrics       # per-fold table + CSV for the Decision Tree
python -m scripts.generate_gini_hand_calculation   # regenerates docs/gini_hand_calculation.md
```

The full three-model comparison (3 models × 2 conditions × 5 folds) runs inside
the notebook's Part 2 section through
`src.evaluation.runner.run_evaluation(inputs)`; the same call backs the
integration test. Fold assignments depend on the pinned `scikit-learn==1.9.1`;
see `docs/decision_tree_results.md` §10 for why the pin matters.

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

## Part 2: hospital-grouped model comparison

Three untuned models — Dummy (`most_frequent`), `GaussianNB()` and
`DecisionTreeClassifier(criterion="gini", random_state=42)` — are evaluated on
**one shared set of five-fold `StratifiedGroupKFold` splits on `HOSPCODE`,
seed 42**, in two feature conditions (`full`, 20 columns; `no_rfv`, 15 columns).
Preprocessing is refitted inside each training fold only, train and test
hospitals are disjoint in every fold, and the primary metric is **Macro-F1**
(fold mean ± sample std), reported alongside pooled out-of-fold metrics over
all 10,207 visits. Implementation: `src/evaluation/`; integration and write-up:
notebook sections P2.1–P2.12.

Headline results, `full` condition (fold mean ± std, scikit-learn 1.9.1):

| Model | Macro-F1 | Accuracy | Note |
|---|---|---|---|
| Dummy (most frequent) | 0.1374 ± 0.0008 | 0.5232 ± 0.0045 | Majority-class floor by construction; ROC-AUC N/A by protocol |
| GaussianNB | 0.1585 ± 0.0069 | 0.1626 ± 0.0074 | Class-3 recall 0.0502; strongly RFV-sensitive |
| Decision Tree (Gini) | **0.2799 ± 0.0198** | 0.4627 ± 0.0075 | Accuracy below the 0.5232 majority-class floor |

Removing RFV lowers tree Macro-F1 to 0.2456 ± 0.0151 and destabilises
GaussianNB (0.0987 ± 0.0507). These are results on a documentation field under
local triage protocols — **not clinical decision support**. Full evidence
(per-fold, pooled, per-class, confusion matrices) and
`reports/figures/fig05_part2_macro_f1_comparison.png` are produced by the
notebook at run time.

## Figures

All five are committed under `reports/figures/` and embedded in the notebook.

| Figure | Shows |
|---|---|
| `fig01_target_distribution.png` | Severe five-class imbalance motivating macro-F1 |
| `fig02_missingness.png` | Missingness by feature, and by target class |
| `fig03_feature_relationships.png` | Acuity by arrival mode and by reason for visit |
| `fig04_hospital_variation.png` | Hospital-to-hospital variation motivating grouped validation |
| `fig05_part2_macro_f1_comparison.png` | Part 2 Macro-F1 by model and condition (fold mean ± std) |

## Git

| Branch | Purpose |
|---|---|
| `main` | Stable. Reviewed work only. |
| `dev` | Active development. All work happens here. |

Raw data is excluded by `.gitignore` on both branches. Work is integrated into
`dev` through reviewed pull requests; no one pushes directly to `dev`.

Merged PR history on `dev`:

| PR | Issue | Contribution |
|---|---|---|
| #6 | #1 | Shared hospital-grouped evaluation framework |
| #8 | #3 | Gini Decision Tree experiment and report |
| #9 | #4 | Hand-worked Gini split calculation |
| #10 | #2 | GaussianNB experiment and report |

PR #7 (an early Part 2 evaluation plan) was closed without merge: it predates
the trained models and its status claims are superseded by the merged framework
above.

## Status

**Assignment 1, Part 1 — complete. Part 2 — models, evaluation and integration
complete. Parts 3–5 — not started.**

| # | Item | Status |
|---|---|---|
| 1 | Instructor dataset approval | **Outstanding** — stated precondition in the course brief |
| 2 | Target formulation and success metric | **Settled** — five-class `IMMEDR`, macro-F1 |
| 3 | Demographic-imputation policy | **Settled** — excluded as target-encoded leakage |
| 4 | NCHS confirmation on redistribution | **Outstanding** — mitigated by git-ignoring the raw file |
| 5 | Feature register | **Settled** — 18 raw → 20 engineered |
| 6 | Leakage-free preprocessing pipeline | **Implemented and verified** |
| 7 | Classifiers and performance measurement (Dummy, GaussianNB, Decision Tree) | **Implemented — Part 2; no hyperparameter tuning** |
| 8 | Hospital-grouped validation (5 folds, seed 42, `HOSPCODE`) | **Implemented — Part 2** |
| 9 | Part 2 comparison write-up, Macro-F1 figure, notebook integration | **Complete — notebook sections P2.1–P2.12** |
| 10 | Part 3: nested tuning, RF/AdaBoost, XGBoost, elastic-net logistic regression | **Not started** |
| 11 | Part 4: MLP, PCA/K-means, interpretability | **Not started** |
| 12 | Part 5: final integrated notebook, report, presentation | **Not started** |

## Academic integrity

The course brief treats uncredited use of code from the internet or AI tools as
a violation. AI-assisted work in this repository is expected to be attributed in
the notebook.
