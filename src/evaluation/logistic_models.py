"""Regularized multinomial logistic regression specifications for Part 3."""

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src import config
from src.evaluation.nested import NestedSpec

LOGISTIC_MODEL_IDS = ("ridge_logistic", "lasso_logistic", "elastic_net_logistic")
SOLVER_SETTINGS = {
    "solver": "saga",
    "random_state": config.RANDOM_STATE,
    "max_iter": 15000,
    "tol": 5e-3,
}


def make_logistic_spec(model_id: str) -> NestedSpec:
    """Build a separate inner search for each family using sklearn 1.9.1."""
    ratios = {
        "ridge_logistic": (0.0,),
        "lasso_logistic": (1.0,),
        "elastic_net_logistic": (0.1, 0.5, 0.9),
    }[model_id]

    def estimator_factory() -> Pipeline:
        # Keep the shared feature values and store the one-hot design sparsely.
        return make_pipeline(
            FunctionTransformer(csr_matrix, accept_sparse=True),
            LogisticRegression(l1_ratio=ratios[0], class_weight=None, **SOLVER_SETTINGS),
        )

    grid = {
        "estimator__logisticregression__C": (0.01, 0.1, 1.0, 10.0, 100.0),
    }
    if len(ratios) > 1:
        grid["estimator__logisticregression__l1_ratio"] = ratios
    return NestedSpec(
        name=model_id,
        estimator_factory=estimator_factory,
        param_grid=grid,
    )
