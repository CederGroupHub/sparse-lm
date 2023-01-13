"""Sanity checks: literally just copied from sklearn tests... """

import pytest
import numpy.testing as npt
import numpy as np
from sklearn.preprocessing import add_dummy_feature

from sparselm.model import OrdinaryLeastSquares


def test_linear_regression():
    # Test LinearRegression on a simple dataset.
    # a simple dataset
    X = [[1], [2]]
    Y = [1, 2]

    reg = OrdinaryLeastSquares()
    reg.fit(X, Y)

    npt.assert_array_almost_equal(reg.coef_, [1])
    npt.assert_array_almost_equal(reg.intercept_, [0])
    npt.assert_array_almost_equal(reg.predict(X), [1, 2])

    # test it also for degenerate input
    X = [[1]]
    Y = [0]

    reg = OrdinaryLeastSquares()
    reg.fit(X, Y)
    npt.assert_array_almost_equal(reg.coef_, [0])
    npt.assert_array_almost_equal(reg.intercept_, [0])
    npt.assert_array_almost_equal(reg.predict(X), [0])


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_sample_weights(fit_intercept, rng):
    # It would not work with under-determined systems
    n_samples, n_features = 6, 5

    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=n_samples)

    sample_weight = 1.0 + rng.uniform(size=n_samples)

    # OLS with explicit sample_weight
    reg = OrdinaryLeastSquares(fit_intercept=fit_intercept)
    reg.fit(X, y, sample_weight=sample_weight)
    coefs1 = reg.coef_
    inter1 = reg.intercept_

    assert reg.coef_.shape == (X.shape[1],)  # sanity checks

    # Closed form of the weighted least square
    # theta = (X^T W X)^(-1) @ X^T W y
    W = np.diag(sample_weight)
    X_aug = X if not fit_intercept else add_dummy_feature(X)

    Xw = X_aug.T @ W @ X_aug
    yw = X_aug.T @ W @ y
    coefs2 = np.linalg.solve(Xw, yw)

    if not fit_intercept:
        npt.assert_allclose(coefs1, coefs2)
    else:
        npt.assert_allclose(coefs1, coefs2[1:])
        npt.assert_allclose(inter1, coefs2[0])
