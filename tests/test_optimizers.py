import pytest
import numpy as np

from sparselm.optimizer import GridSearch, LineSearch
from sparselm.model.miqp.regularized_l0 import L2L0, L1L0

ALL_CRITERION = ["max_r2", "one_std_r2"]
# Currently we will only test on mixedL0
ALL_ESTIMATORS = [L2L0, L1L0]


@pytest.fixture(scope="module")
def param_grid():
    # Test on multiple grids
    return [{"alpha": [0.01, 0.1], "l0_ratio": [0.1, 0.3]},
            {"alpha": [0.02, 0.2], "l0_ratio": [0.2, 0.4]}]


@pytest.fixture(scope="module", params=ALL_ESTIMATORS)
def estimator(request):
    return request.param()


@pytest.fixture(scope="module", params=ALL_CRITERION)
def grid_search(estimator, param_grid, request):
    grid_searcher = GridSearch(estimator, param_grid)
    grid_searcher.opt_selection = request.param
    return grid_searcher


@pytest.fixture(scope="module", params=ALL_CRITERION)
def line_search(estimator, param_grid, request):
    # Multi-grids not supported in line search mode.
    param_grid_lines = sorted([(key, values)
                               for key, values in param_grid[0].items()])
    line_searcher = LineSearch(estimator, param_grid_lines, n_iter=3)
    line_searcher.opt_selection_methods = request.param
    return line_searcher


def test_grid_search(random_model, grid_search):
    femat, energies, ecis = random_model
    grid_search.fit(X=femat, y=energies)
    assert "best_params_" in vars(grid_search)
    best_params = grid_search.best_params_
    assert "alpha" in best_params and "l0_ratio" in best_params
    assert best_params["alpha"] in [0.01, 0.1, 0.02, 0.2]
    assert best_params["l0_ratio"] in [0.1, 0.2, 0.3, 0.4]

    assert 0 <= grid_search.best_score_ <= 1
    assert "coef_" in vars(grid_search.best_estimator_)
    assert "intercept_" in vars(grid_search.best_estimator_)
    energies_pred = grid_search.predict(femat)
    assert np.sum((energies - energies_pred) ** 2) / len(energies) <= 1E-2


def test_line_search(random_model, line_search):
    femat, energies, ecis = random_model
    line_search.fit(X=femat, y=energies)
    assert "best_params_" in vars(line_search)
    assert "alpha" in line_search.best_params_ and "l0_ratio" in line_search.best_params_
    assert 0 <= line_search.best_score_ <= 1
    assert "coef_" in vars(line_search.best_estimator_)
    assert "intercept_" in vars(line_search.best_estimator_)
    energies_pred = line_search.predict(femat)
    assert np.sum((energies - energies_pred) ** 2) / len(energies) <= 1E-2
