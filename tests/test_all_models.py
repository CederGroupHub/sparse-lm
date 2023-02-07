"""General tests for all linear models.

Simply check that they execute successfully on random data.
"""

from inspect import getmembers, isclass, signature

import numpy as np
import pytest

import sparselm.model as spm

ALL_ESTIMATORS = getmembers(spm, isclass)


@pytest.mark.parametrize("estimator_cls", ALL_ESTIMATORS)
def test_general_fit(estimator_cls, random_model, rng):
    print(f"\nGeneral test of {estimator_cls[0]}.")
    X, y, beta = random_model

    # instantiate the estimator
    sig = signature(estimator_cls[1])

    # check for necessary parameters
    args = {}
    if "groups" in sig.parameters:
        args["groups"] = rng.integers(0, 5, size=len(beta))
    if "group_list" in sig.parameters:
        args["group_list"] = [
            rng.choice(range(5), replace=False, size=rng.integers(1, 5))
            for _ in range(len(beta))
        ]
    if "sparse_bound" in sig.parameters:
        args["sparse_bound"] = 12

    estimator = estimator_cls[1](**args)
    estimator.fit(X, y)
    # assert a value of coefficients has been set correctly
    assert isinstance(estimator.coef_, np.ndarray)
    assert len(estimator.coef_) == len(beta)
    assert len(estimator.predict(X)) == len(y)
    assert estimator.intercept_ == 0.0

    estimator = estimator_cls[1](fit_intercept=True, **args)
    estimator.fit(X, y)
    # assert a value of coefficients has been set correctly
    assert isinstance(estimator.coef_, np.ndarray)
    assert len(estimator.coef_) == len(beta)
    assert len(estimator.predict(X)) == len(y)
    assert estimator.intercept_ != 0.0
