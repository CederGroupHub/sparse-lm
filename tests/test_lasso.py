import numpy as np
import numpy.testing as npt
import pytest
from cvxpy.error import SolverError

from sparselm.model import (
    AdaptiveGroupLasso,
    AdaptiveLasso,
    AdaptiveOverlapGroupLasso,
    AdaptiveRidgedGroupLasso,
    AdaptiveSparseGroupLasso,
    GroupLasso,
    Lasso,
    OverlapGroupLasso,
    SparseGroupLasso,
)

ADAPTIVE_ESTIMATORS = [
    AdaptiveLasso,
    AdaptiveGroupLasso,
    AdaptiveSparseGroupLasso,
    AdaptiveOverlapGroupLasso,
    AdaptiveRidgedGroupLasso,
]

THRESHOLD = 1e-8  # relative threshold


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

    lasso = Lasso(alpha=1.0)
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
    X, y, _ = random_model
    lasso = Lasso(fit_intercept=True)
    alasso = AdaptiveLasso(fit_intercept=True)

    lasso.fit(X, y)
    alasso.fit(X, y)

    assert sum(abs(lasso.coef_) > THRESHOLD) >= sum(abs(alasso.coef_) > THRESHOLD)


# TODO flakey test, depends on THRESHOLD value
@pytest.mark.xfail(raises=SolverError)
@pytest.mark.parametrize(
    "standardize",
    [True, False],
)  # standardize=False leads to failures
def test_group_lasso(random_model_with_groups, solver, standardize):
    X, y, _, groups = random_model_with_groups

    aglasso = AdaptiveGroupLasso(
        groups=groups,
        alpha=0.1,
        fit_intercept=True,
        standardize=standardize,
        solver=solver,
    )
    aglasso.fit(X, y)

    # check that if all coefs in groups are consistent
    for gid in np.unique(groups):
        m = np.max(abs(aglasso.coef_))
        all_active = (abs(aglasso.coef_[groups == gid]) > m * THRESHOLD).all()
        all_inactive = (abs(aglasso.coef_[groups == gid]) <= m * THRESHOLD).all()
        assert all_active or all_inactive


@pytest.mark.xfail(raises=SolverError)
@pytest.mark.parametrize(
    "standardize",
    [True, False],
)
def test_group_lasso_weights(random_model_with_groups, solver, standardize):
    X, y, _, groups = random_model_with_groups

    group_weights = np.ones(len(np.unique(groups)))

    aglasso = AdaptiveGroupLasso(
        groups=groups,
        alpha=0.1,
        group_weights=group_weights,
        fit_intercept=True,
        standardize=standardize,
        solver=solver,
    )
    aglasso.fit(X, y)

    rglasso = AdaptiveRidgedGroupLasso(
        groups=groups,
        alpha=0.1,
        group_weights=group_weights,
        fit_intercept=True,
        standardize=standardize,
        solver=solver,
    )
    rglasso.fit(X, y)

    # check that if all coefs in groups are consistent
    for gid in np.unique(groups):
        m = np.max(abs(aglasso.coef_))

        all_active = (abs(aglasso.coef_[groups == gid]) > m * THRESHOLD).all()
        all_inactive = (abs(aglasso.coef_[groups == gid]) <= m * THRESHOLD).all()
        assert all_active or all_inactive

        m = np.max(abs(rglasso.coef_))
        all_active = (abs(rglasso.coef_[groups == gid]) > m * THRESHOLD).all()
        all_inactive = (abs(rglasso.coef_[groups == gid]) <= m * THRESHOLD).all()
        assert all_active or all_inactive


@pytest.mark.xfail(raises=SolverError)
@pytest.mark.parametrize("estimator_cls", ADAPTIVE_ESTIMATORS)
def test_adaptive_weights(estimator_cls, random_model_with_groups, solver, rng):
    X, y, beta, groups = random_model_with_groups

    if estimator_cls.__name__ == "AdaptiveLasso":
        estimator = estimator_cls(solver=solver)
    elif estimator_cls.__name__ == "AdaptiveOverlapGroupLasso":
        gids = np.unique(groups)
        group_list = [
            rng.choice(gids, replace=False, size=rng.integers(1, 3))
            for _ in range(len(beta))
        ]
        estimator = estimator_cls(group_list=group_list, solver=solver)
    else:
        estimator = estimator_cls(groups=groups, solver=solver)

    # force generating weights
    estimator.generate_problem(X, y)

    if estimator_cls.__name__ == "AdaptiveSparseGroupLasso":
        weights = [
            estimator.canonicals_.parameters.adaptive_coef_weights.value.copy(),
            estimator.canonicals_.parameters.adaptive_group_weights.value.copy(),
        ]
    else:
        weights = [estimator.canonicals_.parameters.adaptive_weights.value.copy()]

    estimator.fit(X, y)

    if estimator_cls.__name__ == "AdaptiveSparseGroupLasso":
        new_weights = [
            estimator.canonicals_.parameters.adaptive_coef_weights.value.copy(),
            estimator.canonicals_.parameters.adaptive_group_weights.value.copy(),
        ]
    else:
        new_weights = [estimator.canonicals_.parameters.adaptive_weights.value.copy()]

    # simply check that the weights are updated.
    # TODO a better check would be to check that weights for active groups/coefs
    #  are smaller than those of inactive ones
    for nw, w in zip(new_weights, weights):
        assert not any(nw_i == pytest.approx(w_i) for nw_i, w_i in zip(nw, w))


def test_bad_inputs(random_model_with_groups, rng):
    X, y, beta, groups = random_model_with_groups
    bad_groups = rng.integers(0, 6, size=len(beta) - 1)
    group_weights = np.ones(len(np.unique(bad_groups)))

    # test that warns when no groups given
    with pytest.warns(UserWarning):
        gl = GroupLasso()
        gl.fit(X, y)

    with pytest.warns(UserWarning):
        gl = OverlapGroupLasso()
        gl.fit(X, y)

    # bad groups
    with pytest.raises(ValueError):
        gl = GroupLasso(bad_groups, group_weights=group_weights)
        gl.fit(X, y)

    with pytest.raises(TypeError):
        gl = GroupLasso("groups", group_weights=group_weights)
        gl.fit(X, y)

    # bad group_weights
    with pytest.raises(ValueError):
        group_weights = np.ones(len(np.unique(bad_groups)) - 1)
        gl = GroupLasso(bad_groups, group_weights=group_weights)
        gl.fit(X, y)

    with pytest.raises(TypeError):
        gl = GroupLasso(groups, group_weights="weights")
        gl.fit(X, y)

    # bad l1_ratio
    lasso = SparseGroupLasso(groups)
    with pytest.raises(ValueError):
        lasso.l1_ratio = -1.0
        lasso.fit(X, y)

    with pytest.raises(ValueError):
        lasso.l1_ratio = 2.0
        lasso.fit(X, y)

    with pytest.raises(ValueError):
        sgl = SparseGroupLasso(groups, l1_ratio=-1.0)
        sgl.fit(X, y)

    with pytest.raises(ValueError):
        sgl = SparseGroupLasso(groups, l1_ratio=2.0)
        sgl.fit(X, y)

    # test that it warns
    with pytest.warns(UserWarning):
        sgl = SparseGroupLasso(groups, l1_ratio=0.0)
        sgl.fit(X, y)
    with pytest.warns(UserWarning):
        sgl = SparseGroupLasso(groups, l1_ratio=1.0)
        sgl.fit(X, y)


@pytest.mark.parametrize("estimator_cls", ADAPTIVE_ESTIMATORS)
def test_set_parameters(estimator_cls, random_model_with_groups, rng):
    X, y, beta, groups = random_model_with_groups

    if estimator_cls.__name__ == "AdaptiveLasso":
        estimator = estimator_cls()
    elif estimator_cls.__name__ == "AdaptiveOverlapGroupLasso":
        gids = np.unique(groups)
        group_list = [
            rng.choice(gids, replace=False, size=rng.integers(1, 3))
            for _ in range(len(beta))
        ]
        estimator = estimator_cls(group_list=group_list)
    else:
        estimator = estimator_cls(groups=groups)

    estimator.alpha = 0.5
    assert estimator.alpha == 0.5
    estimator.generate_problem(X, y)
    assert estimator.canonicals_.parameters.alpha.value == 0.5

    if hasattr(estimator, "l1_ratio"):
        # default l1_ratio is 0.5
        assert estimator.canonicals_.parameters.lambda1.value == 0.5 * 0.5
        assert estimator.canonicals_.parameters.lambda2.value == 0.5 * 0.5

        estimator.l1_ratio = 0.25
        estimator._set_param_values()
        assert estimator.l1_ratio == 0.25
        assert estimator.canonicals_.parameters.lambda1.value == 0.25 * 0.5
        assert estimator.canonicals_.parameters.lambda2.value == 0.75 * 0.5

    if hasattr(estimator, "delta"):
        estimator.delta = (4.0,)
        estimator._set_param_values()
        npt.assert_array_equal(
            estimator.canonicals_.parameters.delta.value,
            4.0 * np.ones(len(np.unique(groups))),
        )

        estimator.delta = 3.0 * np.ones(len(np.unique(groups)))
        estimator._set_param_values()
        npt.assert_array_equal(estimator.delta, 3.0 * np.ones(len(np.unique(groups))))
        npt.assert_array_equal(
            estimator.canonicals_.parameters.delta.value,
            3.0 * np.ones(len(np.unique(groups))),
        )
