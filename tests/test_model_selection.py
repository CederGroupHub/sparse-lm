import cvxpy as cp
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, train_test_split

from sparselm.model import L1L0, L2L0
from sparselm.model_selection import GridSearchCV, LineSearchCV

ALL_CRITERION = ["max_score", "one_std_score"]
# Currently we will only test on mixedL0
ALL_ESTIMATORS = [L2L0, L1L0]
ONLY_L2L0 = [L2L0]


@pytest.fixture(scope="module")
def param_grid():
    # Test on multiple grids
    return [
        {"alpha": [0.01, 0.1], "eta": [0.03, 0.3]},
        {"alpha": [0.02, 0.2], "eta": [0.04, 0.4]},
    ]


def test_solver():
    # Check that your solvers can work well.
    # Non-academic, non-commercial Gurobi can not solve large scale model > 100 params.
    # ECOS_BB is significantly slower, so use gurobi if possible!
    x = cp.Variable(10, integer=True)
    obj = cp.sum_squares(x)
    cons = [x <= 3, x >= -3]
    prob = cp.Problem(cp.Minimize(obj), cons)

    if "GUROBI" in cp.installed_solvers():
        result = prob.solve(solver="GUROBI")
    else:
        result = prob.solve(solver="ECOS_BB")

    assert x.value is not None
    assert result is not None


@pytest.fixture(scope="module", params=ALL_ESTIMATORS)
def estimator(random_energy_model, request):
    ecis = random_energy_model[2]
    # Each correlation function as its own group. Doing ordinary hierarchy.
    groups = list(range(len(ecis)))
    if "GUROBI" in cp.installed_solvers():
        return request.param(groups=groups, solver="GUROBI")
    else:
        return request.param(groups=groups, solver="ECOS_BB")
    # return request.param(solver="ECOS_BB")


@pytest.fixture(scope="module", params=ONLY_L2L0)
def mixed_l2l0_est(random_energy_model, request):
    ecis = random_energy_model[2]
    # Each correlation function as its own group. Doing ordinary hierarchy.
    groups = list(range(len(ecis)))
    if "GUROBI" in cp.installed_solvers():
        return request.param(groups=groups, solver="GUROBI")
    else:
        return request.param(groups=groups, solver="ECOS_BB")
    # return request.param(solver="ECOS_BB")


def test_mixed_l0_wts(random_energy_model, mixed_l2l0_est, rng):
    femat, energies, _ = random_energy_model
    mixed_l2l0_est.eta = 1e-5
    mixed_l2l0_est.fit(X=femat, y=energies)
    energies_pred = mixed_l2l0_est.predict(femat)
    assert energies_pred is not None
    mixed_l2l0_est.tikhonov_w = 1000 * rng.random(femat.shape[1])
    mixed_l2l0_est.fit(X=femat, y=energies)
    energies_pred_wtd = mixed_l2l0_est.predict(femat)
    assert energies_pred_wtd is not None


@pytest.fixture(scope="module", params=ALL_CRITERION)
def grid_search(estimator, param_grid, request):
    grid_searcher = GridSearchCV(
        estimator, param_grid, opt_selection_method=request.param
    )
    return grid_searcher


@pytest.fixture(scope="module", params=ALL_CRITERION)
def line_search(estimator, param_grid, request):
    # Multi-grids not supported in line search mode.
    param_grid_lines = sorted((key, values) for key, values in param_grid[0].items())
    line_searcher = LineSearchCV(
        estimator,
        param_grid_lines,
        opt_selection_method=request.param,
        n_iter=3,
    )
    return line_searcher


def test_grid_search(random_energy_model, grid_search):
    femat, energies, _ = random_energy_model
    n_samples, n_features = femat.shape
    grid_search.fit(X=femat, y=energies)
    assert "best_params_" in vars(grid_search)
    best_params = grid_search.best_params_
    assert "alpha" in best_params and "eta" in best_params
    assert best_params["alpha"] in [0.01, 0.1, 0.02, 0.2]
    assert best_params["eta"] in [0.03, 0.3, 0.04, 0.4]

    assert grid_search.best_score_ <= 1
    assert "coef_" in vars(grid_search.best_estimator_)
    assert "intercept_" in vars(grid_search.best_estimator_)
    energies_pred = grid_search.predict(femat)
    rmse = np.sum((energies - energies_pred) ** 2) / len(energies)
    # Overfit.
    if n_samples < n_features:
        assert -grid_search.best_score_ >= rmse


# Guarantees that one-std rule always select larger params than max score.
def test_onestd():
    success = 0
    for _ in range(10):
        X, y, coef = make_regression(
            n_samples=200,
            n_features=100,
            n_informative=10,
            noise=40.0,
            bias=-15.0,
            coef=True,
            random_state=0,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )

        # create estimators
        lasso = Lasso(fit_intercept=True)

        # create cv search objects for each estimator
        cv5 = KFold(n_splits=5, shuffle=True, random_state=0)
        params = {"alpha": np.logspace(-1, 1, 10)}

        lasso_cv_std = GridSearchCV(
            lasso, params, opt_selection_method="one_std_score", cv=cv5, n_jobs=-1
        )
        lasso_cv_opt = GridSearchCV(
            lasso, params, opt_selection_method="max_score", cv=cv5, n_jobs=-1
        )

        # fit models on training data
        lasso_cv_std.fit(X_train, y_train)
        lasso_cv_opt.fit(X_train, y_train)

        correct_params = (
            lasso_cv_opt.best_params_["alpha"] <= lasso_cv_std.best_params_["alpha"]
        )
        sparsity_opt = np.sum(np.abs(lasso_cv_opt.best_estimator_.coef_) >= 1e-6)
        sparsity_std = np.sum(np.abs(lasso_cv_std.best_estimator_.coef_) >= 1e-6)

        if correct_params and sparsity_opt >= sparsity_std:
            success += 1

    # Allow some failure caused by randomness of CV splits.
    assert success >= 8


def test_line_search(random_energy_model, line_search):
    femat, energies, _ = random_energy_model
    n_samples, n_features = femat.shape
    line_search.fit(X=femat, y=energies)
    assert "best_params_" in vars(line_search)
    best_params = line_search.best_params_
    assert "alpha" in best_params and "eta" in best_params
    assert best_params["alpha"] in [0.01, 0.1]
    assert best_params["eta"] in [0.03, 0.3]

    assert line_search.best_score_ <= 1
    assert "coef_" in vars(line_search.best_estimator_)
    assert "intercept_" in vars(line_search.best_estimator_)
    energies_pred = line_search.predict(femat)
    rmse = np.sum((energies - energies_pred) ** 2) / len(energies)
    # Overfit.
    if n_samples < n_features:
        assert -line_search.best_score_ >= rmse
