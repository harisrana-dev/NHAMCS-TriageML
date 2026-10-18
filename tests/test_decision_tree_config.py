"""Configuration checks for the Gini Decision Tree (Issue #3).

AI-assisted: drafted with Claude, reviewed and run by the author.
"""

from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation.runner import (
    MODEL_IDS,
    build_estimator_pipeline,
    make_estimator,
)

MODEL_ID = "decision_tree_gini"


def _expected_params() -> dict:
    return DecisionTreeClassifier(
        criterion="gini", random_state=42
    ).get_params()


def test_model_id_is_registered() -> None:
    assert MODEL_ID in MODEL_IDS


def test_estimator_is_gini_tree_with_seed_42() -> None:
    estimator = make_estimator(MODEL_ID)
    assert isinstance(estimator, DecisionTreeClassifier)
    assert estimator.criterion == "gini"
    assert estimator.random_state == 42
    assert estimator.random_state == config.RANDOM_STATE


def test_no_extra_parameters_are_set() -> None:
    # Only criterion and random_state differ from sklearn defaults, so no
    # depth limit or other tuning was added.
    estimator = make_estimator(MODEL_ID)
    assert estimator.get_params() == _expected_params()
    assert estimator.max_depth is None


def test_factory_returns_a_fresh_estimator_each_call() -> None:
    assert make_estimator(MODEL_ID) is not make_estimator(MODEL_ID)


def test_pipeline_uses_the_same_configuration() -> None:
    pipeline = build_estimator_pipeline(MODEL_ID)
    estimator = pipeline.named_steps["estimator"]
    assert isinstance(estimator, DecisionTreeClassifier)
    assert estimator.get_params() == _expected_params()
