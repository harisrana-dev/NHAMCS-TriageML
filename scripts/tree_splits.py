"""Print fold-0 tree splits from the shared evaluator run (Issue #3).

The pipelines are retained from ``run_evaluation``; no additional folds or
tree fits are created here.

AI-assisted: drafted with Claude, reviewed and run by the author.
"""

from sklearn.tree import export_text

from src.data import dataset, target
from src.evaluation import prepare_modelling_inputs
from src.evaluation.data import FEATURE_CONDITIONS
from src.evaluation.runner import run_evaluation

FOLD_ID = 0
TOP_DEPTH = 3


def main() -> None:
    full = dataset.load_raw_dataset(verify=True, expect_shape=True)
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)
    result = run_evaluation(
        inputs,
        model_ids=("decision_tree_gini",),
        retain_fitted_pipelines=True,
    )
    fold = next((fold for fold in result.folds if fold.fold_id == FOLD_ID), None)
    if fold is None:
        raise ValueError(f"Fold {FOLD_ID} is not present in the evaluator result.")

    print(
        "fold", fold.fold_id,
        "n_train", fold.train_positions.size,
        "n_test", fold.test_positions.size,
    )

    for condition in FEATURE_CONDITIONS:
        pipe = result.fitted_pipelines[
            ("decision_tree_gini", condition, fold.fold_id)
        ]
        tree = pipe.named_steps["estimator"]
        names = list(pipe[:-1].get_feature_names_out())

        print(f"\n##### condition: {condition} #####")
        print("depth:", tree.get_depth(), "leaves:", tree.get_n_leaves())
        print("top 10 feature importances:")
        ranked = sorted(
            zip(names, tree.feature_importances_), key=lambda t: -t[1]
        )
        for name, value in ranked[:10]:
            print(f"  {value:.3f}  {name}")
        print(f"\nexport_text (max_depth={TOP_DEPTH}):")
        print(export_text(tree, feature_names=names, max_depth=TOP_DEPTH))


if __name__ == "__main__":
    main()
