"""Run Ridge, Lasso, and Elastic Net through the Part 3 nested harness."""

from dataclasses import replace
from importlib.metadata import version
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from scripts.run_ensemble_nested import write_results
from src import config
from src.data import dataset, target
from src.evaluation import LABELS, prepare_modelling_inputs, run_nested_evaluation
from src.evaluation.logistic_models import (
    LOGISTIC_MODEL_IDS,
    SOLVER_SETTINGS,
    make_logistic_spec,
)
from src.evaluation.data import ModellingInputs
from src.evaluation.nested import (
    DEFAULT_INNER_SPLITS,
    DEFAULT_OUTER_SPLITS,
    NestedEvaluationResult,
)
from src.evaluation.splits import make_grouped_folds

OUTPUT_DIR = config.REPO_ROOT / "reports" / "logistic_nested"
REPRODUCTION_COMMAND = "python -m scripts.run_logistic_nested"


def with_oof_probabilities(result: NestedEvaluationResult) -> NestedEvaluationResult:
    """Export the harness's class-aligned probabilities in OOF row order."""
    oof = result.oof_predictions.copy()
    probabilities = np.full((len(oof), len(LABELS)), np.nan)
    for fold in result.fold_results:
        probabilities[fold.outer_test_positions] = fold.outer_probabilities
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError(f"Invalid or incomplete OOF probabilities for {result.model}.")
    oof[[f"proba_{label}" for label in LABELS]] = probabilities
    return replace(result, oof_predictions=oof)


def run_logistic_experiment(
    inputs: ModellingInputs, *, verbose: int = 0
) -> list[NestedEvaluationResult]:
    """Use common outer folds and abort on any nonconverged fit."""
    folds = make_grouped_folds(
        inputs.y, inputs.groups, random_state=config.RANDOM_STATE
    )
    specs = [make_logistic_spec(model_id) for model_id in LOGISTIC_MODEL_IDS]
    results = []
    # The harness fits sequentially in this process, including outer refits.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for spec in specs:
            if verbose:
                print(spec.name, flush=True)
            result = run_nested_evaluation(
                inputs,
                spec,
                outer_folds=folds,
                random_state=config.RANDOM_STATE,
                search_n_jobs=1,
                error_score="raise",
                verbose=verbose,
            )
            results.append(with_oof_probabilities(result))
    return results


def main() -> None:
    data_path = config.resolve_dataset_path()
    raw = dataset.load_raw_dataset(path=data_path, verify=True, expect_shape=True)
    integrity = dataset.verify_dataset(data_path)
    inputs = prepare_modelling_inputs(target.select_modelling_population(raw))
    results = run_logistic_experiment(inputs, verbose=1)
    runs = {}
    for result in results:
        spec = make_logistic_spec(result.model)
        estimator = spec.estimator_factory().named_steps["logisticregression"]
        runs[f"{result.model}/{result.condition}"] = {
            **result.metadata,
            "estimator_params": estimator.get_params(),
            "param_grid": spec.param_grid,
            "feature_columns": list(result.feature_columns),
        }
    metadata = {
        "reproduction_command": REPRODUCTION_COMMAND,
        "dataset": {
            "filename": integrity["filename"],
            "size_bytes": integrity["size_bytes"],
            "sha256": integrity["sha256"],
            "shape": list(raw.shape),
            "cohort_rows": inputs.n_rows,
            "cohort_hospitals": inputs.groups.nunique(),
        },
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in (
                    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"
                )
            },
        },
        "protocol": {
            "splitter": "StratifiedGroupKFold",
            "shuffle": True,
            "groups": "HOSPCODE",
            "target": "IMMEDR",
            "seed": config.RANDOM_STATE,
            "outer_folds": DEFAULT_OUTER_SPLITS,
            "inner_folds": DEFAULT_INNER_SPLITS,
            "selection_metric": "mean inner macro-F1, labels 1–5, zero_division=0",
            "candidate_tie_break": "first maximum in ParameterGrid order",
            "probability_class_order": list(LABELS),
            "convergence_warning": "error",
        },
        "solver_settings": SOLVER_SETTINGS,
        "runs": runs,
    }
    write_results(results, inputs, metadata=metadata, output_dir=OUTPUT_DIR)
    print(pd.concat([result.summary for result in results]).to_string(index=False))
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
