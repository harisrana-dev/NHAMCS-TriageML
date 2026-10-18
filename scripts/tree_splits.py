"""Print trained-tree splits for Issue #3 (one shared fold).

AI-assisted: drafted with Claude, reviewed and run by the author.
"""

from sklearn.tree import export_text

from src.data import dataset, target
from src.evaluation import prepare_modelling_inputs
from src.evaluation.data import (
    FEATURE_CONDITIONS,
    feature_columns_for,
    select_features,
)
from src.evaluation.runner import build_estimator_pipeline
from src.evaluation.splits import make_grouped_folds

FOLD_ID = 0
TOP_DEPTH = 3


def main() -> None:
    full = dataset.load_raw_dataset(verify=True, expect_shape=True)
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)
    folds = make_grouped_folds(
        inputs.y, inputs.groups, n_splits=5, random_state=None
    )
    fold = folds[FOLD_ID]
    print(
        "fold", fold.fold_id,
        "n_train", fold.train_positions.size,
        "n_test", fold.test_positions.size,
    )

    for condition in FEATURE_CONDITIONS:
        columns = feature_columns_for(condition)
        X = select_features(inputs, condition)
        pipe = build_estimator_pipeline("decision_tree_gini", columns=columns)
        pipe.fit(
            X.iloc[fold.train_positions], inputs.y.iloc[fold.train_positions]
        )
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
