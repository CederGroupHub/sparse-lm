"""Test composite estimator class."""
import numpy as np
import numpy.testing as npt
import pytest
from sklearn.base import clone
from sklearn.utils._param_validation import InvalidParameterError

from sparselm.model import L2L0, Lasso
from sparselm.model_selection import GridSearchCV
from sparselm.stepwise import StepwiseEstimator


def test_make_composite():
    # Test making a composite estimator.
    lasso1 = Lasso(fit_intercept=True, alpha=1.0)
    lasso2 = Lasso(fit_intercept=False, alpha=2.0)
    l2l0 = L2L0(groups=[0, 0, 1, 2], alpha=0.1, eta=4.0)
    steps = [("lasso1", lasso1), ("lasso2", lasso2), ("l2l0", l2l0)]

    scope1 = [0, 1, 8]
    scope2 = [2, 3]
    scope3 = [4, 5, 6, 7]
    estimator = StepwiseEstimator(steps, [scope1, scope2, scope3])
    # sklearn convention tests, need pandas.
    # Currently, not passing because conventional sklearn estimator should not have
    # fixed number of features.
    # check_estimator(estimator)
    assert estimator.steps[0][1].fit_intercept
    assert not estimator.steps[1][1].fit_intercept
    assert not estimator.steps[2][1].fit_intercept

    # check parameters. Nested estimator case not tested yet.
    params = estimator.get_params(deep=True)
    assert params["lasso1"].get_params(deep=True)["alpha"] == 1.0
    assert params["lasso2"].get_params(deep=True)["alpha"] == 2.0
    assert params["l2l0"].get_params(deep=True)["alpha"] == 0.1
    assert params["l2l0"].get_params(deep=True)["eta"] == 4.0
    assert params["lasso1__alpha"] == 1.0
    assert params["lasso2__alpha"] == 2.0
    assert params["l2l0__alpha"] == 0.1
    assert params["l2l0__eta"] == 4.0

    estimator.set_params(lasso2__alpha=0.5, l2l0__alpha=0.2, l2l0__eta=3.0)
    params = estimator.get_params(deep=True)
    assert params["lasso1"].get_params(deep=True)["alpha"] == 1.0
    assert params["lasso2"].get_params(deep=True)["alpha"] == 0.5
    assert params["l2l0"].get_params(deep=True)["alpha"] == 0.2
    assert params["l2l0"].get_params(deep=True)["eta"] == 3.0
    assert params["lasso1__alpha"] == 1.0
    assert params["lasso2__alpha"] == 0.5
    assert params["l2l0__alpha"] == 0.2
    assert params["l2l0__eta"] == 3.0

    # Test unsafe clone, such that composite can be used in the optimizers.
    # Currently, have to mute sanity check from origianl sklearn clone.
    cloned = clone(estimator)
    params = cloned.get_params(deep=True)
    assert params["lasso1"].get_params(deep=True)["alpha"] == 1.0
    assert params["lasso2"].get_params(deep=True)["alpha"] == 0.5
    assert params["l2l0"].get_params(deep=True)["alpha"] == 0.2
    assert params["l2l0"].get_params(deep=True)["eta"] == 3.0
    assert params["lasso1__alpha"] == 1.0
    assert params["lasso2__alpha"] == 0.5
    assert params["l2l0__alpha"] == 0.2
    assert params["l2l0__eta"] == 3.0

    # A searcher can also be put into stepwise.
    grid = GridSearchCV(lasso2, {"alpha": [0.01, 0.1, 1.0]})
    steps = [("lasso1", lasso1), ("lasso2", grid), ("l2l0", l2l0)]
    estimator = StepwiseEstimator(steps, [scope1, scope2, scope3])
    # check_estimator(estimator)
    params = estimator.get_params(deep=True)
    assert params["lasso1__alpha"] == 1.0
    assert params["l2l0__alpha"] == 0.2
    assert params["l2l0__eta"] == 3.0
    assert "lasso2__alpha" not in params
    assert params["lasso2__estimator__alpha"] == 0.5


def test_toy_composite():
    lasso1 = Lasso(fit_intercept=True, alpha=1e-6)
    lasso2 = Lasso(fit_intercept=False, alpha=1e-6)
    grid = GridSearchCV(clone(lasso2), {"alpha": [1e-8, 1e-7, 1e-6]})
    bad_lasso2 = Lasso(fit_intercept=True, alpha=1e-6)
    l2l0 = L2L0(groups=[0, 0, 1, 2], alpha=0, eta=1e-9)
    steps = [("lasso1", lasso1), ("lasso2", lasso2), ("l2l0", l2l0)]
    steps2 = [("lasso1", clone(lasso1)), ("lasso2", grid), ("l2l0", clone(l2l0))]
    bad_steps = [("lasso1", lasso1), ("lasso2", bad_lasso2), ("l2l0", l2l0)]

    scope1 = [0, 1, 8]
    scope2 = [2, 3]
    scope3 = [4, 5, 6, 7]
    estimator = StepwiseEstimator(steps, [scope1, scope2, scope3])
    # Use grid search on lasso2.
    estimator2 = StepwiseEstimator(steps2, [scope1, scope2, scope3])

    bad_scope1 = [0, 1]
    bad_scope2 = [3, 4]
    bad_scope3 = [5, 6, 7, 8]
    bad_estimator1 = StepwiseEstimator(steps, [bad_scope1, bad_scope2, bad_scope3])
    bad_estimator2 = StepwiseEstimator(bad_steps, [scope1, scope2, scope3])

    w_test = np.random.normal(scale=2, size=9) * 0.2
    w_test[0] = 10
    w_test[-1] = 0.5
    # A bad feature matrix with too many features.
    bad_X = np.random.random(size=(20, 12))
    bad_X[:, 0] = 1
    with pytest.raises(ValueError):
        estimator.fit(bad_X, np.random.random(size=20))
    X = np.random.random(size=(20, 9))
    X[:, 0] = 1
    X[:, -1] = -8 * np.random.random(size=20)
    y = np.dot(X, w_test) + np.random.normal(scale=0.01, size=20)

    # Bad scopes.
    with pytest.raises(InvalidParameterError):
        bad_estimator1.fit(X, y)
    # Allow fit intercept in beyond the first estimator.
    with pytest.raises(InvalidParameterError):
        bad_estimator2.fit(X, y)
    # A correct estimator.

    def run_estimator_test(estimator_test):
        estimator_test.fit(X, y)
        # print("intercept:", estimator_test.intercept_)
        # print("coef:", estimator_test.coef_)

        assert estimator_test.intercept_ == estimator_test.steps[0][1].intercept_
        assert not np.any(np.isnan(estimator_test.coef_))

        assert not np.isclose(estimator_test.intercept_, 0)

        for (_, sub), scope in zip(
            estimator_test.steps, estimator_test.estimator_feature_indices
        ):
            if hasattr(sub, "estimator"):
                sub_coef = sub.best_estimator_.coef_
            else:
                sub_coef = sub.coef_
            npt.assert_array_almost_equal(sub_coef, estimator_test.coef_[scope])
        coef_1 = estimator_test.coef_.copy()
        intercept_1 = estimator_test.intercept_

        # Now do not fit intercept.
        estimator_test.steps[0][1].fit_intercept = False
        estimator_test.fit(X, y)
        coef_2 = estimator_test.coef_.copy()
        intercept_2 = estimator_test.intercept_
        assert np.isclose(intercept_2, 0)

        # Do some naive assertion on the fitted coefficients.
        assert abs(coef_1[0] + intercept_1 - 10) / 10 <= 0.1
        assert abs(coef_2[0] - 10) / 10 <= 0.1
        # assert np.linalg.norm(coef_2 - w_test) / np.linalg.norm(w_test) <= 0.4

        total_y = np.zeros(len(y))
        for (_, sub_estimator_test), sub_scope in zip(
            estimator_test.steps, estimator_test.estimator_feature_indices
        ):
            total_y += sub_estimator_test.predict(X[:, sub_scope])
        npt.assert_array_almost_equal(estimator_test.predict(X), total_y)
        npt.assert_array_almost_equal(
            np.dot(X, estimator_test.coef_) + estimator_test.intercept_, total_y
        )

    # Either estimators should be able to work.
    run_estimator_test(estimator)
    run_estimator_test(estimator2)
