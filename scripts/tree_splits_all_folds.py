"""Summarize fitted Gini trees from the shared evaluator run (Issue #3).

This script retains and inspects the pipelines fitted by ``run_evaluation``;
it does not generate folds or fit a second set of trees.

AI-assisted: drafted with Claude, reviewed and run by the author.
"""

import pandas as pd
from sklearn.tree import export_text

from src.data import dataset, target
from src.evaluation import prepare_modelling_inputs
from src.evaluation.data import FEATURE_CONDITIONS
from src.evaluation.runner import run_evaluation

TOP_DEPTH = 3
TOP_N = 10

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 60)


def describe_split(tree, names, node):
    t = tree.tree_
    if t.children_left[node] == -1:
        return "(leaf)"
    return f"{names[t.feature[node]]} <= {t.threshold[node]:.2f}"


def main() -> None:
    full = dataset.load_raw_dataset(verify=True, expect_shape=True)
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)
    result = run_evaluation(
        inputs,
        model_ids=("decision_tree_gini",),
        retain_fitted_pipelines=True,
    )
    folds = result.folds
    rows = []
    top_features = {}
    text_parts = []

    for condition in FEATURE_CONDITIONS:
        for fold in folds:
            pipe = result.fitted_pipelines[
                ("decision_tree_gini", condition, fold.fold_id)
            ]
            tree = pipe.named_steps["estimator"]
            names = list(pipe[:-1].get_feature_names_out())
            t = tree.tree_

            ranked = sorted(
                zip(names, tree.feature_importances_), key=lambda p: -p[1]
            )[:TOP_N]
            top_features[(condition, fold.fold_id)] = [n for n, _ in ranked]

            fold_metric = result.fold_metrics.loc[
                (result.fold_metrics["model"] == "decision_tree_gini")
                & (result.fold_metrics["condition"] == condition)
                & (result.fold_metrics["fold"] == fold.fold_id),
                "macro_f1",
            ]
            if len(fold_metric) != 1:
                raise RuntimeError(
                    f"Expected one evaluator metric row for {condition}/"
                    f"fold {fold.fold_id}, got {len(fold_metric)}."
                )

            rows.append(
                {
                    "condition": condition,
                    "fold": fold.fold_id,
                    "n_train": int(fold.train_positions.size),
                    "depth": int(tree.get_depth()),
                    "leaves": int(tree.get_n_leaves()),
                    "train_rows_per_leaf": round(
                        fold.train_positions.size / tree.get_n_leaves(), 2
                    ),
                    "root_split": describe_split(tree, names, 0),
                    "left_child": describe_split(
                        tree, names, t.children_left[0]
                    ),
                    "right_child": describe_split(
                        tree, names, t.children_right[0]
                    ),
                    "top1": ranked[0][0],
                    "top1_importance": round(float(ranked[0][1]), 3),
                    "heldout_macro_f1": float(fold_metric.iloc[0]),
                }
            )
            text_parts.append(
                f"##### {condition} / fold {fold.fold_id} #####\n"
                f"depth {tree.get_depth()} leaves {tree.get_n_leaves()}\n"
                + export_text(tree, feature_names=names, max_depth=TOP_DEPTH)
            )

    table = pd.DataFrame(rows)
    print("=== Tree structure per fold ===")
    print(
        table[
            ["condition", "fold", "n_train", "depth", "leaves",
             "train_rows_per_leaf", "heldout_macro_f1"]
        ].to_string(index=False)
    )
    print("\n=== Root split and its two children per fold ===")
    print(
        table[
            ["condition", "fold", "root_split", "left_child", "right_child"]
        ].to_string(index=False)
    )

    for condition in FEATURE_CONDITIONS:
        print(f"\n=== Top {TOP_N} importances per fold: {condition} ===")
        sets = []
        for fold in folds:
            feats = top_features[(condition, fold.fold_id)]
            sets.append(set(feats))
            print(f"fold {fold.fold_id}: " + ", ".join(feats[:5]) + ", ...")
        common = sorted(set.intersection(*sets))
        print(f"in the top {TOP_N} of ALL 5 folds ({len(common)}):")
        for name in common:
            print("   ", name)

    table.to_csv("docs/decision_tree_all_folds_tree_summary.csv", index=False)
    with open("docs/decision_tree_all_folds_splits.txt", "w") as handle:
        handle.write("\n\n".join(text_parts) + "\n")
    print(
        "\nsaved: docs/decision_tree_all_folds_tree_summary.csv, "
        "docs/decision_tree_all_folds_splits.txt"
    )


if __name__ == "__main__":
    main()
