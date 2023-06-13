"""General tests for all linear models.

Simply check that they execute successfully on random data.
"""

from inspect import getmembers, isclass, signature

import cvxpy as cp
import numpy as np
import pytest
from cvxpy.error import SolverError
from sklearn.utils.estimator_checks import check_estimator

import sparselm.model as spm
from sparselm.model._miqp._base import MIQPl0

ESTIMATORS = getmembers(spm, isclass)
ESTIMATOR_NAMES = [est[0] for est in ESTIMATORS]
ESTIMATORS = [est[1] for est in ESTIMATORS]


@pytest.fixture(params=ESTIMATORS, ids=ESTIMATOR_NAMES)
def estimator(request):
    estimator_cls = request.param
    if issubclass(estimator_cls, MIQPl0):
        regressor = estimator_cls(fit_intercept=True, solver="SCIP")
        if hasattr(regressor, "eta"):
            regressor.eta = 0.01
        return regressor
    return estimator_cls(fit_intercept=True, solver="ECOS")


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_general_fit(estimator_cls, random_model, rng):
    X, y, beta = random_model

    # instantiate the estimator
    sig = signature(estimator_cls)

    # check for necessary parameters
    args = {}
    if "groups" in sig.parameters:
        args["groups"] = rng.integers(0, 5, size=len(beta))
    if "group_list" in sig.parameters:
        args["group_list"] = [
            np.sort(rng.choice(range(5), replace=False, size=rng.integers(1, 5)))
            for _ in range(len(beta))
        ]
    if "sparse_bound" in sig.parameters:
        args["sparse_bound"] = 12

    estimator = estimator_cls(**args)
    estimator.fit(X, y)
    # assert a value of coefficients has been set correctly
    assert isinstance(estimator.coef_, np.ndarray)
    assert len(estimator.coef_) == len(beta)
    assert len(estimator.predict(X)) == len(y)
    assert estimator.intercept_ == 0.0

    estimator = estimator_cls(fit_intercept=True, **args)
    estimator.fit(X, y)
    # assert a value of coefficients has been set correctly
    assert isinstance(estimator.coef_, np.ndarray)
    assert len(estimator.coef_) == len(beta)
    assert len(estimator.predict(X)) == len(y)
    assert estimator.intercept_ != 0.0


@pytest.mark.xfail(raises=SolverError)
def test_add_constraints(estimator, random_model, rng):
    with pytest.raises(RuntimeError):
        estimator.add_constraints([cp.Variable(1) >= 0])

    X, y, beta = random_model
    estimator.generate_problem(X, y)
    n_constraints = len(estimator.canonicals_.constraints)
    # a dummy constraint
    estimator.add_constraints([estimator.canonicals_.beta >= 0.0])
    assert len(estimator.canonicals_.problem.constraints) == n_constraints + 1
    assert len(estimator.canonicals_.user_constraints) == 1
    assert len(estimator.canonicals_.constraints) == n_constraints

    # force cache data
    # ( solving the model sometimes fails and we only want to check that a warning is
    # raised )
    estimator.cached_X_ = X
    estimator.cached_y_ = y

    new_X = rng.random(X.shape)
    with pytest.warns(UserWarning):
        estimator.fit(new_X, y)


def test_sklearn_compatible(estimator):
    """Test sklearn compatibility with no parameter inputs."""
    check_estimator(estimator)
