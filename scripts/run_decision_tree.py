"""Run the Gini Decision Tree on the shared evaluator (Issue #3).

AI-assisted: script drafted with Claude, reviewed and run by the author.
"""

import pandas as pd

from src.data import dataset, target
from src.evaluation import prepare_modelling_inputs
from src.evaluation.runner import run_evaluation

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def main() -> None:
    # 1. Inputs: same three steps as the integration test
    full = dataset.load_raw_dataset(verify=True, expect_shape=True)
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)
    print("rows:", inputs.n_rows, "hospitals:", inputs.n_hospitals)

    # 2. Shared folds, both conditions (full, no_rfv), tree only
    result = run_evaluation(inputs, model_ids=("decision_tree_gini",))

    # 3. Results
    print("\n=== Fold diagnostics ===")
    print(result.diagnostics)

    print("\n=== Summary (mean/std over folds) ===")
    cols = [
        "model", "condition", "n_folds",
        "macro_f1_mean", "macro_f1_std",
        "balanced_accuracy_mean", "accuracy_mean",
        "roc_auc_mean", "roc_auc_std",
    ]
    print(result.summary[cols])

    print("\n=== Pooled out-of-fold metrics ===")
    print(result.pooled_oof_metrics)

    print("\n=== Per-class metrics ===")
    print(result.per_class_metrics)

    for (model, condition), frame in result.confusion_matrices.items():
        print(f"\n=== Confusion matrix: {model} / {condition} ===")
        print(frame)


if __name__ == "__main__":
    main()
