"""Per-fold metrics for the Gini Decision Tree (Issue #3).

AI-assisted: drafted with Claude, reviewed and run by the author.
"""

import pandas as pd
from sklearn.metrics import f1_score

from src.data import dataset, target
from src.evaluation import prepare_modelling_inputs
from src.evaluation.data import LABELS
from src.evaluation.runner import run_evaluation

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def main() -> None:
    full = dataset.load_raw_dataset(verify=True, expect_shape=True)
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)
    result = run_evaluation(inputs, model_ids=("decision_tree_gini",))

    labels = [int(label) for label in LABELS]
    rows = []
    oof = result.oof_predictions
    for (condition, fold), grp in oof.groupby(["condition", "fold"]):
        rows.append(
            {
                "condition": condition,
                "fold": fold,
                "micro_f1": f1_score(
                    grp["y_true"], grp["y_pred"], labels=labels, average="micro"
                ),
                "weighted_f1": f1_score(
                    grp["y_true"], grp["y_pred"], labels=labels,
                    average="weighted", zero_division=0,
                ),
            }
        )
    extra = pd.DataFrame(rows)

    main_cols = [
        "condition", "fold", "n_test", "macro_f1", "accuracy",
        "balanced_accuracy", "roc_auc",
    ]
    class_cols = [f"f1_{label}" for label in LABELS]
    table = result.fold_metrics[main_cols + class_cols].merge(
        extra, on=["condition", "fold"]
    )
    table = table.sort_values(["condition", "fold"])
    table["micro_equals_accuracy"] = (
        (table["micro_f1"] - table["accuracy"]).abs() < 1e-9
    )

    first = [
        "condition", "fold", "n_test", "macro_f1", "micro_f1",
        "weighted_f1", "accuracy", "balanced_accuracy", "roc_auc",
        "micro_equals_accuracy",
    ]
    print("=== Per-fold metrics ===")
    print(table[first].round(4).to_string(index=False))
    print("\n=== Per-fold per-class F1 ===")
    print(
        table[["condition", "fold"] + class_cols].round(4).to_string(index=False)
    )

    num = ["macro_f1", "micro_f1", "weighted_f1", "accuracy",
           "balanced_accuracy", "roc_auc"]
    print("\n=== Mean and std over folds ===")
    print(table.groupby("condition")[num].agg(["mean", "std"]).round(4))

    table.to_csv("docs/decision_tree_fold_metrics.csv", index=False)
    print("\nsaved: docs/decision_tree_fold_metrics.csv")


if __name__ == "__main__":
    main()
