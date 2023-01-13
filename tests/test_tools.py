from functools import partial
import numpy.testing as npt
from sparselm.model import OrdinaryLeastSquares
from sparselm.tools import constrain_coefficients


# TODO repeat test upon warning that new coefficients are not within range
def test_constrain_coefficients(rng):
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
    cstr_coefs = constrain_coefficients(inds, 2, 0)(partial(fit, reg=reg))(X, y)

    assert cstr_coefs.shape == coefs.shape
    for i in inds:
        assert 0 <= cstr_coefs[i] <= 2

    @constrain_coefficients(inds, 2, 0)
    def fit_constrained(X, y, reg):
        reg.fit(X, y)
        return reg.coef_

    cstr_coefs2 = fit_constrained(X, y, reg=reg)
    npt.assert_almost_equal(cstr_coefs, cstr_coefs2)

    # Test different low and high values
    low = rng.random(size=3) -0.5
    high = rng.random(size=3) + low
    cstr_coefs = constrain_coefficients(inds, high, low)(partial(fit, reg=reg))(X, y)

    assert cstr_coefs.shape == coefs.shape
    for i, l, h in zip(inds, low, high):
        assert l <= cstr_coefs[i] <= h

    @constrain_coefficients(inds, high, low)
    def fit_constrained(X, y, reg):
        reg.fit(X, y)
        return reg.coef_

    cstr_coefs2 = fit_constrained(X, y, reg=reg)
    npt.assert_almost_equal(cstr_coefs, cstr_coefs2)
