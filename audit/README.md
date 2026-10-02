# Audit

## The forensic audit lives outside this repository

The acquisition, integrity verification and forensic audit of the NHAMCS 2022
Emergency Department public-use file were performed in a **separate workspace**:

```
C:\Projects\ML_Dataset_Audit\NHAMCS_2022\
```

That workspace owns the raw data, the official NCHS documentation and the
read-only audit scripts. Nothing in it is copied here. This repository
**consumes its verified conclusions** rather than duplicating the forensic
workspace.

Deliberately **not** copied into this repository:

| Not copied | Reason |
|---|---|
| `raw/nhamcs_ed_2022/ed2022-stata.dta` | Raw data must not be redistributed (see [`../data/README.md`](../data/README.md)) |
| `raw/nhamcs_ed_2022/ed2022-stata.zip` | Contains the raw file |
| `raw/WRONG_DATASET_NHCS/nhcs2022ed_stata.dta` | Wrong survey (NHCS, not NHAMCS); quarantined for provenance |
| `documentation/` (PDF + Stata/SAS readmes, `.dct`, `.do`) | Not ours to redistribute; each is obtainable from CDC/NCHS |
| `audit/variable_inventory.csv`, `leakage_register.csv` | 913-row and 799-row generated tables; regenerable from the audit scripts |
| `audit/scripts/01`–`15` | Belong to the audit workspace, which stays read-only |

## What this repository carries forward instead

Summary artefacts only, with each claim traceable to a named source file in the
audit workspace.

| Topic | Consumed conclusion | Source in audit workspace |
|---|---|---|
| Dataset identity | `ed2022-stata.dta`, 37,720,094 bytes, SHA-256 `0E00CD5A…C881C2`, 16,025 × 913 | `verification/verification_report.txt` |
| Verdict | `CONDITIONAL CANDIDATE` — six preconditions before any modelling | `audit/audit_report.md` §14 |
| Target | `IMMEDR`, 5-level ordinal, "immediacy with which patient should be seen (unimputed)" | `audit_report.md` §3 |
| Modelling population | `IMMEDR ∈ {1..5}` → 10,207 rows | `audit_report.md` §3.2 |
| Feature eligibility | All 913 variables classified A–G/T; 35 candidates (13 A + 22 B) | `audit/variable_inventory.csv`, `audit/feature_eligibility.csv` |
| Leakage | 794 class C + 5 class D excluded; post-target by construction | `audit/leakage_register.csv` |
| Circular imputation | NCHS imputed `SEX`, `AGE`, `RACER`, `ETHIM` using the target; 22.84% of the modelling population affected | `audit_report.md` §12, `audit/imputation_leakage_exposure.csv` |
| Validation | `StratifiedGroupKFold(5, groups="HOSPCODE")`; 147 hospitals; random split invalid | `audit_report.md` §9, `audit/group_structure.csv` |
| Course compliance | Mapping to the CS-13410 brief; dataset approval outstanding | `audit/course_compliance_matrix.md` |

The per-variable eligibility register is **restated** in
[`../docs/feature_register.md`](../docs/feature_register.md) so this repository
is self-contained on the decision that matters. The full 913-row table stays in
the audit workspace.

## Referencing the audit

When a finding is used in the notebook or report, cite the audit artefact, not
this summary. Example:

> Feature availability at the prediction point is classified per-variable in
> `ML_Dataset_Audit/NHAMCS_2022/audit/variable_inventory.csv`
> (`audit_report.md` §5).

## Re-verification

If the raw file is ever replaced, the audit must be re-run before the analysis
is trusted:

1. Recompute the SHA-256 and confirm it matches the pinned value.
2. Re-run the audit workspace's `audit/scripts/01`–`15`.
3. Re-check every figure quoted in this repository against the regenerated
   outputs.

The audit is only as good as that hash.
