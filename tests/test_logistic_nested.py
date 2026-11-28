"""Checks for the Part 3 regularized logistic experiment."""

from collections import Counter

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from scripts.run_logistic_nested import run_logistic_experiment, with_oof_probabilities
from src.evaluation import LABELS, run_nested_evaluation
from src.evaluation.logistic_models import LOGISTIC_MODEL_IDS, make_logistic_spec
from src.evaluation.splits import make_grouped_folds


@pytest.mark.parametrize(
    "model_id, ratios, count",
    [
        ("ridge_logistic", {0.0}, 5),
        ("lasso_logistic", {1.0}, 5),
        ("elastic_net_logistic", {0.1, 0.5, 0.9}, 15),
    ],
)
def test_logistic_search_space(model_id, ratios, count):
    spec = make_logistic_spec(model_id)
    estimator = spec.estimator_factory().named_steps["logisticregression"]
    assert isinstance(estimator, LogisticRegression)
    assert estimator.solver == "saga"
    candidates = list(ParameterGrid(spec.param_grid))
    assert len(candidates) == count
    assert {row["estimator__logisticregression__C"] for row in candidates} == {0.01, 0.1, 1, 10, 100}
    assert estimator.class_weight is None
    assert "estimator__logisticregression__class_weight" not in spec.param_grid
    assert {
        row.get("estimator__logisticregression__l1_ratio", estimator.l1_ratio) for row in candidates
    } == ratios
    assert ("estimator__logisticregression__l1_ratio" in spec.param_grid) == (len(ratios) > 1)


@pytest.mark.parametrize("model_id", LOGISTIC_MODEL_IDS)
def test_logistic_fold_local_fits_and_oof(model_id, synthetic_inputs, monkeypatch):
    inputs = synthetic_inputs
    folds = make_grouped_folds(inputs.y, inputs.groups)
    spec = make_logistic_spec(model_id)
    fit_sizes = []
    original_fit = StandardScaler.fit

    def record_fit(self, X, y=None, sample_weight=None):
        fit_sizes.append(len(X))
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    result = with_oof_probabilities(
        run_nested_evaluation(
            inputs, spec, outer_folds=folds, retain_fitted_pipelines=True
        )
    )
    expected_sizes = []
    oof = result.oof_predictions
    assert oof["row_index"].tolist() == list(inputs.X.index)
    assert oof["row_index"].is_unique
    np.testing.assert_array_equal(oof["y_true"], inputs.y)
    assert result.best_params["fold"].tolist() == list(range(5))
    assert set(result.best_params["model"]) == {model_id}
    assert result.best_params["best_inner_score"].notna().all()
    assert "HOSPCODE" not in result.feature_columns
    assert "IMMEDR" not in result.feature_columns

    for outer, fold in zip(folds, result.fold_results, strict=True):
        assert not outer.train_hospitals & outer.test_hospitals
        expected_sizes.append(len(outer.train_positions))
        for inner in fold.inner_folds:
            assert not inner.train_hospitals & inner.val_hospitals
            assert inner.train_hospitals | inner.val_hospitals == outer.train_hospitals
            expected_sizes.extend(
                [len(inner.train_positions)] * len(ParameterGrid(spec.param_grid))
            )
        test = oof.iloc[outer.test_positions]
        assert (test["fold"] == outer.fold_id).all()
        pipeline = result.fitted_pipelines[outer.fold_id]
        np.testing.assert_array_equal(pipeline.classes_, LABELS)
        probabilities = test[[f"proba_{label}" for label in LABELS]].to_numpy()
        np.testing.assert_allclose(
            probabilities, pipeline.predict_proba(inputs.X.iloc[outer.test_positions])
        )
        np.testing.assert_array_equal(test["y_pred"], np.asarray(LABELS)[probabilities.argmax(axis=1)])

    assert Counter(fit_sizes) == Counter(expected_sizes)
    assert result.pooled_oof_metrics.iloc[0]["macro_f1"] == pytest.approx(
        f1_score(oof["y_true"], oof["y_pred"], labels=LABELS, average="macro", zero_division=0)
    )


@pytest.mark.parametrize("fail_at", ["inner", "refit"])
def test_convergence_warning_aborts_experiment(synthetic_inputs, monkeypatch, fail_at):
    import warnings

    original_fit = LogisticRegression.fit
    outer = make_grouped_folds(synthetic_inputs.y, synthetic_inputs.groups)[0]

    def warn_on_partition(self, X, y, sample_weight=None):
        is_refit = X.shape[0] == len(outer.train_positions)
        if is_refit == (fail_at == "refit"):
            warnings.warn("test convergence failure", ConvergenceWarning)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", warn_on_partition)
    if fail_at == "inner":
        with pytest.raises(RuntimeError, match="Inner fit failed") as error:
            run_logistic_experiment(synthetic_inputs)
        assert isinstance(error.value.__cause__, ConvergenceWarning)
    else:
        with pytest.raises(ConvergenceWarning, match="test convergence failure"):
            run_logistic_experiment(synthetic_inputs)
