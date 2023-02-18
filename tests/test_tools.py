import warnings
from functools import partial

import numpy.testing as npt
import pytest

from sparselm.model import OrdinaryLeastSquares
from sparselm.tools import constrain_coefficients


@pytest.mark.parametrize("test_number", range(5))  # run the test 5 times
def test_constrain_coefficients(test_number, rng):
    n_samples, n_features = 10, 8
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=n_samples)
    reg = OrdinaryLeastSquares(fit_intercept=True)
    reg.fit(X, y)
    coefs = reg.coef_

    def fit(X, y, reg):
        reg.fit(X, y)
        return reg.coef_

    # Test uniform low and high values
    inds = rng.choice(n_features, size=3, replace=False)

    with warnings.catch_warnings(record=True) as w:
        cstr_coefs = constrain_coefficients(inds, 2, 0)(partial(fit, reg=reg))(X, y)

    assert cstr_coefs.shape == coefs.shape

    # Check if warning was raised, meaning coefficients were not within range
    # in that case just test that the indeed that warning was raised.
    if len(w) > 0:
        with pytest.warns(RuntimeWarning):
            cstr_coefs = constrain_coefficients(inds, 2, 0)(partial(fit, reg=reg))(X, y)
    else:
        for i in inds:
            assert 0 <= cstr_coefs[i] <= 2

    @constrain_coefficients(inds, 2, 0)
    def fit_constrained1(X, y, reg):
        reg.fit(X, y)
        return reg.coef_

    cstr_coefs2 = fit_constrained1(X, y, reg=reg)
    npt.assert_almost_equal(cstr_coefs, cstr_coefs2)

    # Test different low and high values
    low = rng.random(size=3) - 0.5
    high = rng.random(size=3) + low

    with warnings.catch_warnings(record=True) as w:
        cstr_coefs = constrain_coefficients(inds, high, low)(partial(fit, reg=reg))(
            X, y
        )

    assert cstr_coefs.shape == coefs.shape

    # Check if warning was raised, meaning coefficients were not within range
    # in that case just test that the indeed that warning was raised.
    if len(w) > 0:
        with pytest.warns(RuntimeWarning):
            cstr_coefs = constrain_coefficients(inds, high, low)(partial(fit, reg=reg))(
                X, y
            )
    else:
        for i, l, h in zip(inds, low, high):
            assert l <= cstr_coefs[i] <= h

    @constrain_coefficients(inds, high, low)
    def fit_constrained2(X, y, reg):
        reg.fit(X, y)
        return reg.coef_

    cstr_coefs2 = fit_constrained2(X, y, reg=reg)
    npt.assert_almost_equal(cstr_coefs, cstr_coefs2)

    # just use high value
    with warnings.catch_warnings(record=True) as w:
        cstr_coefs = constrain_coefficients(inds, high=high)(partial(fit, reg=reg))(
            X, y
        )

    assert cstr_coefs.shape == coefs.shape

    # Check if warning was raised, meaning coefficients were not within range
    # in that case just test that the indeed that warning was raised.
    if len(w) > 0:
        with pytest.warns(RuntimeWarning):
            cstr_coefs = constrain_coefficients(inds, high=high)(partial(fit, reg=reg))(
                X, y
            )
    else:
        for i, h in zip(inds, high):
            assert cstr_coefs[i] <= h

    # just use low value
    with warnings.catch_warnings(record=True) as w:
        cstr_coefs = constrain_coefficients(inds, low=low)(partial(fit, reg=reg))(X, y)

    assert cstr_coefs.shape == coefs.shape

    # Check if warning was raised, meaning coefficients were not within range
    # in that case just test that the indeed that warning was raised.
    if len(w) > 0:
        with pytest.warns(RuntimeWarning):
            cstr_coefs = constrain_coefficients(inds, low=low)(partial(fit, reg=reg))(
                X, y
            )
    else:
        for i, l in zip(inds, low):
            assert l <= cstr_coefs[i]


# TODO write this test
def test_r2_score_to_cv_error():
    pass
