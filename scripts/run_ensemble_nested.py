"""Run nested hospital-grouped Part 3 ensemble experiments."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix

from src import config
from src.data import dataset, target
from src.evaluation import (
    ENSEMBLE_MODEL_IDS,
    LABELS,
    make_ensemble_spec,
    prepare_modelling_inputs,
    run_nested_evaluation,
    select_full_condition_winner,
)
from src.evaluation.splits import fold_diagnostics

OUTPUT_DIR = config.REPO_ROOT / "reports" / "ensemble_nested"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _confusion_rows(results, inputs) -> pd.DataFrame:
    frames = []
    for result in results:
        fold_frames = []
        for fold in result.folds:
            positions = fold.test_positions
            matrix = confusion_matrix(
                inputs.y.iloc[positions],
                result.fold_results[fold.fold_id].outer_predictions,
                labels=list(LABELS),
            )
            frame = pd.DataFrame(
                matrix,
                index=pd.Index(LABELS, name="true"),
                columns=pd.Index(LABELS, name="predicted"),
            ).stack().rename("count").reset_index()
            frame.insert(0, "fold", fold.fold_id)
            frame.insert(0, "scope", "outer_fold")
            fold_frames.append(frame)
        pooled = (
            result.confusion_matrix.rename_axis(index="true", columns="predicted")
            .stack()
            .rename("count")
            .reset_index()
        )
        pooled.insert(0, "fold", "")
        pooled.insert(0, "scope", "pooled_oof")
        for frame in [*fold_frames, pooled]:
            frame.insert(0, "condition", result.condition)
            frame.insert(0, "model", result.model)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_results(
    results, inputs, *, metadata: dict[str, Any], output_dir: Path = OUTPUT_DIR
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "fold_metrics": [result.fold_metrics for result in results],
        "summary": [result.summary for result in results],
        "pooled_oof_metrics": [result.pooled_oof_metrics for result in results],
        "oof_predictions": [result.oof_predictions for result in results],
        "per_class_metrics": [result.per_class_metrics for result in results],
        "best_params": [result.best_params for result in results],
        "inner_search": [result.inner_search for result in results],
    }
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(
            output_dir / f"{name}.csv",
            index=False,
            float_format="%.10g",
        )
    pooled_class_rows = []
    for result in results:
        per_class = result.pooled_oof_metrics.iloc[0]["per_class"]
        for label, scores in per_class.items():
            pooled_class_rows.append(
                {
                    "model": result.model,
                    "condition": result.condition,
                    "class": int(label),
                    **scores,
                }
            )
    pd.DataFrame(pooled_class_rows).to_csv(
        output_dir / "pooled_per_class_metrics.csv",
        index=False,
        float_format="%.10g",
    )
    diagnostics = []
    for result in results:
        frame = result.diagnostics.copy()
        frame.insert(0, "condition", result.condition)
        frame.insert(0, "model", result.model)
        diagnostics.append(frame)
    pd.concat(diagnostics, ignore_index=True).to_csv(
        output_dir / "nested_fold_diagnostics.csv",
        index=False,
    )
    outer_diagnostics = []
    for result in results:
        frame = fold_diagnostics(result.folds)
        frame.insert(0, "condition", result.condition)
        frame.insert(0, "model", result.model)
        outer_diagnostics.append(frame)
    pd.concat(outer_diagnostics, ignore_index=True).to_csv(
        output_dir / "outer_fold_diagnostics.csv",
        index=False,
    )
    _confusion_rows(results, inputs).to_csv(
        output_dir / "confusion_matrices.csv",
        index=False,
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    data_path = config.resolve_dataset_path()
    integrity = dataset.verify_dataset(data_path)
    if not integrity["verified"]:
        raise ValueError(f"Dataset integrity check failed for {data_path}.")
    raw = dataset.load_raw_dataset(path=data_path, verify=True, expect_shape=True)
    inputs = prepare_modelling_inputs(target.select_modelling_population(raw))

    full_results = [
        run_nested_evaluation(
            inputs,
            make_ensemble_spec(model_id, condition="full"),
            verbose=1,
        )
        for model_id in ENSEMBLE_MODEL_IDS
    ]
    full_by_model = {result.model: result for result in full_results}
    inner_scores = {
        model_id: result.best_params["best_inner_score"].astype(float).tolist()
        for model_id, result in full_by_model.items()
    }
    family_winner = select_full_condition_winner(inner_scores)
    no_rfv_result = run_nested_evaluation(
        inputs,
        make_ensemble_spec(family_winner, condition="no_rfv"),
        verbose=1,
    )
    results = [*full_results, no_rfv_result]

    metadata = {
        "protocol": {
            "seed": config.RANDOM_STATE,
            "outer_splitter": "StratifiedGroupKFold",
            "outer_folds": 5,
            "inner_splitter": "StratifiedGroupKFold",
            "inner_folds": 3,
            "groups": "HOSPCODE",
            "selection_metric": "fixed-label macro-F1 (labels 1-5, zero_division=0)",
            "parallelism": 1,
            "family_winner_rule": (
                "For full-condition runs, select the model with the highest "
                "mean across outer folds of its best inner-CV macro-F1; ties "
                "are resolved by ENSEMBLE_MODEL_IDS order. Select no_rfv only "
                "for that predeclared winner."
            ),
            "family_winner_inner_scores": inner_scores,
            "no_rfv_family_winner": family_winner,
        },
        "dataset": {
            "filename": integrity["filename"],
            "size_bytes": integrity["size_bytes"],
            "sha256": integrity["sha256"],
            "shape": list(raw.shape),
        },
        "platform": platform.platform(),
        "runs": {
            result.model + "/" + result.condition: result.metadata
            for result in results
        },
    }
    write_results(results, inputs, metadata=metadata)
    print("\nNested OOF summary:")
    print(
        pd.concat([result.summary for result in results], ignore_index=True)
        .to_string(index=False)
    )
    print(f"\nno_rfv family winner: {family_winner}")
    print(f"Results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
