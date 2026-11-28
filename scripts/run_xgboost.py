"""Run the XGBoost nested grouped-CV experiment (Issue #13)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from src import config
from src.data import dataset, target
from src.evaluation import (
    LABELS,
    INVERSE_FREQUENCY_WEIGHT_POLICY,
    NestedSpec,
    prepare_modelling_inputs,
    run_nested_evaluation,
)


class XGBoostUrgencyClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost adapter preserving the project's original labels 1–5."""

    def __init__(
        self,
        max_depth=3,
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.asarray(LABELS)
        label_to_index = {
            label: index for index, label in enumerate(self.classes_)
        }
        y_encoded = np.asarray([label_to_index[int(v)] for v in y])

        self.model_ = XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABELS),
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
            random_state=self.random_state,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
        )
        self.model_.fit(
            X,
            y_encoded,
            sample_weight=sample_weight,
        )
        return self

    def predict(self, X):
        encoded = self.model_.predict(X).astype(int)
        return self.classes_[encoded]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def main():
    print("Loading dataset...")
    full = dataset.load_raw_dataset(verify=True, expect_shape=True)
    population = target.select_modelling_population(full)
    inputs = prepare_modelling_inputs(population)

    print(
        f"Rows: {inputs.n_rows:,} | "
        f"Hospitals: {inputs.n_hospitals:,} | "
        f"XGBoost: {xgboost.__version__}"
    )

    spec = NestedSpec(
        name="xgboost",
        estimator_factory=lambda: XGBoostUrgencyClassifier(
            random_state=config.RANDOM_STATE
        ),
        param_grid={
            "estimator__max_depth": [3, 5],
        },
        condition="full",
        weight_policies=(
            None,
            INVERSE_FREQUENCY_WEIGHT_POLICY,
        ),
    )

    print("\nStarting nested grouped-CV...")
    result = run_nested_evaluation(
        inputs,
        spec,
        n_outer_splits=5,
        n_inner_splits=3,
        random_state=config.RANDOM_STATE,
        search_n_jobs=1,
        verbose=0,
    )

    output_dir = Path("reports/xgboost_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "fold_metrics.csv": result.fold_metrics,
        "summary.csv": result.summary,
        "oof_predictions.csv": result.oof_predictions,
        "pooled_oof_metrics.csv": result.pooled_oof_metrics,
        "per_class_metrics.csv": result.per_class_metrics,
        "confusion_matrix.csv": result.confusion_matrix,
        "best_params.csv": result.best_params,
        "inner_search.csv": result.inner_search,
        "diagnostics.csv": result.diagnostics,
    }

    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)

    metadata = dict(result.metadata)
    metadata["xgboost_version"] = xgboost.__version__
    metadata["estimator_configuration"] = {
        "objective": "multi:softprob",
        "num_class": len(LABELS),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": 1,
        "tested_max_depth": [3, 5],
        "n_estimators": 100,
        "learning_rate": 0.05,
    }

    with (output_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2, default=str)

    print("\n=== Nested CV summary ===")
    print(result.summary.to_string(index=False))

    print("\n=== Selected parameters per outer fold ===")
    print(result.best_params.to_string(index=False))

    print("\n=== Pooled out-of-fold metrics ===")
    print(result.pooled_oof_metrics.to_string(index=False))

    print(f"\nAll results saved in: {output_dir}")


if __name__ == "__main__":
    main()
