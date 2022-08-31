"""This file only contains software functionality tests, which means
we only test on randomly generated feature matrices, ecis and energies,
to make sure our codes will run, but the physicality of results are not
checked in real CE systems."""

import cvxpy as cp
import numpy as np
import pytest

from sparselm.model.miqp._regularized_l0 import L1L0, L2L0
from sparselm.optimizer import GridSearch, LineSearch

ALL_CRITERION = ["max_r2", "one_std_r2"]
# Currently we will only test on mixedL0
ALL_ESTIMATORS = [L2L0, L1L0]


@pytest.fixture(scope="module")
def param_grid():
    # Test on multiple grids
    return [
        {"alpha": [0.01, 0.1], "l0_ratio": [0.1, 0.3]},
        {"alpha": [0.02, 0.2], "l0_ratio": [0.2, 0.4]},
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
def estimator(request):
    if "GUROBI" in cp.installed_solvers():
        return request.param(solver="GUROBI")
    else:
        return request.param(solver="ECOS_BB")
    # return request.param(solver="ECOS_BB")


def test_single_estimator(random_model, estimator):
    femat, energies, ecis = random_model
    estimator.fit(X=femat, y=energies)
    energies_pred = estimator.predict(femat)
    assert energies_pred is not None


@pytest.fixture(scope="module", params=ALL_CRITERION)
def grid_search(estimator, param_grid, request):
    grid_searcher = GridSearch(
        estimator, param_grid, opt_selection_method=request.param
    )
    return grid_searcher


@pytest.fixture(scope="module", params=ALL_CRITERION)
def line_search(estimator, param_grid, request):
    # Multi-grids not supported in line search mode.
    param_grid_lines = sorted((key, values) for key, values in param_grid[0].items())
    line_searcher = LineSearch(
        estimator, param_grid_lines, opt_selection_method=request.param, n_iter=3
    )
    return line_searcher


def test_grid_search(random_model, grid_search):
    femat, energies, ecis = random_model
    grid_search.fit(X=femat, y=energies)
    assert "best_params_" in vars(grid_search)
    best_params = grid_search.best_params_
    assert "alpha" in best_params and "l0_ratio" in best_params
    assert best_params["alpha"] in [0.01, 0.1, 0.02, 0.2]
    assert best_params["l0_ratio"] in [0.1, 0.2, 0.3, 0.4]

    assert grid_search.best_score_ <= 1
    assert "coef_" in vars(grid_search.best_estimator_)
    assert "intercept_" in vars(grid_search.best_estimator_)
    energies_pred = grid_search.predict(femat)
    if grid_search.best_score_ > 0.8:
        assert np.sum((energies - energies_pred) ** 2) / len(energies) <= 1e-1


def test_line_search(random_model, line_search):
    femat, energies, ecis = random_model
    line_search.fit(X=femat, y=energies)
    assert "best_params_" in vars(line_search)
    best_params = line_search.best_params_
    assert "alpha" in best_params and "l0_ratio" in best_params
    assert best_params["alpha"] in [0.01, 0.1]
    assert best_params["l0_ratio"] in [0.1, 0.3]

    assert line_search.best_score_ <= 1
    assert "coef_" in vars(line_search.best_estimator_)
    assert "intercept_" in vars(line_search.best_estimator_)
    energies_pred = line_search.predict(femat)
    if line_search.best_score_ > 0.8:
        assert np.sum((energies - energies_pred) ** 2) / len(energies) <= 1e-1
