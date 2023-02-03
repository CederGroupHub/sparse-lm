import numpy as np
import numpy.testing as npt
import pytest

from sparselm.model import  (
    GroupLasso,
    Lasso,
    OverlapGroupLasso,
    RidgedGroupLasso,
    SparseGroupLasso,
)

from sparselm.model import (
    AdaptiveGroupLasso,
    AdaptiveLasso,
    AdaptiveOverlapGroupLasso,
    AdaptiveRidgedGroupLasso,
    AdaptiveSparseGroupLasso,
)


# a high threshold since beta from make_regression are always ~ 1E1
THRESHOLD = 1E-2


@pytest.fixture(params=[4, 10])
def random_model_with_groups(random_model, rng, request):
    """Add a correct set of groups to model."""
    X, y, beta = random_model
    coef_mask = abs(beta) > THRESHOLD
    n_groups = request.param
    n_active_groups = n_groups // 3 + 1
    active_group_inds = rng.choice(range(n_groups), size=n_active_groups, replace=False)
    inactive_group_inds = np.setdiff1d(range(n_groups), active_group_inds)

    groups = np.zeros(len(beta))
    for i, c in enumerate(coef_mask):
        groups[i] = rng.choice(active_group_inds) if c else rng.choice(inactive_group_inds)

    return X, y, beta, groups



def test_lasso_toy():
    # Borrowed from sklearn tests
    # Test Lasso on a toy example for various values of alpha.
    # When validating this against glmnet notice that glmnet divides it
    # against nobs.

    X = [[-1], [0], [1]]
    Y = [-1, 0, 1]  # just a straight line
    T = [[2], [3], [4]]  # test sample

    lasso = Lasso(alpha=1e-8)
    lasso.fit(X, Y)
    pred = lasso.predict(T)
    npt.assert_array_almost_equal(lasso.coef_, [1])
    npt.assert_array_almost_equal(pred, [2, 3, 4])

    lasso = Lasso(alpha=0.1)
    lasso.fit(X, Y)
    pred = lasso.predict(T)
    npt.assert_array_almost_equal(lasso.coef_, [0.85])
    npt.assert_array_almost_equal(pred, [1.7, 2.55, 3.4])

    lasso = Lasso(alpha=0.5)
    lasso.fit(X, Y)
    pred = lasso.predict(T)
    npt.assert_array_almost_equal(lasso.coef_, [0.25])
    npt.assert_array_almost_equal(pred, [0.5, 0.75, 1.0])

    lasso = Lasso(alpha=1)
    lasso.fit(X, Y)
    pred = lasso.predict(T)
    npt.assert_array_almost_equal(lasso.coef_, [0.0])
    npt.assert_array_almost_equal(pred, [0, 0, 0])



def test_lasso_non_float_y():
    # Borrowed from sklearn tests
    X = [[0, 0], [1, 1], [-1, -1]]
    y = [0, 1, 2]
    y_float = [0.0, 1.0, 2.0]

    lasso = Lasso(fit_intercept=False)
    lasso.fit(X, y)
    lasso_float = Lasso(fit_intercept=False)
    lasso_float.fit(X, y_float)
    npt.assert_array_equal(lasso.coef_, lasso_float.coef_)


def test_adaptive_lasso_sparser(random_model):
    X, y, beta = random_model
    lasso = Lasso(fit_intercept=True)
    alasso = AdaptiveLasso(fit_intercept=True)

    lasso.fit(X, y)
    alasso.fit(X, y)

    assert sum(abs(lasso.coef_) > THRESHOLD) >= sum(abs(alasso.coef_) > THRESHOLD)


# TODO flakey test, depends on THRESHOLD value
def test_group_lasso(random_model_with_groups):
    X, y, beta, groups = random_model_with_groups
    active_mask = abs(beta) > THRESHOLD

    glasso = GroupLasso(groups=groups, alpha=5, fit_intercept=True)
    glasso.fit(X, y)

    aglasso = AdaptiveGroupLasso(groups=groups, alpha=0.1, fit_intercept=True)
    aglasso.fit(X, y)

    # check that if all coefs in groups are consistent
    for gid in np.unique(groups):
        assert (abs(glasso.coef_[groups == gid]) > THRESHOLD).all() or (abs(glasso.coef_[groups == gid]) <= THRESHOLD).all()
        assert (abs(aglasso.coef_[groups == gid]) > THRESHOLD).all() or (abs(aglasso.coef_[groups == gid]) <= THRESHOLD).all()

