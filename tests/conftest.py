import numpy as np
import pytest
from sklearn.datasets import make_regression, make_sparse_coded_signal

SEED = 0

# A few solvers to test for convex problems
# ECOS sometimes fails for Adaptive group estimators, but is fast
# SCS and CXVOPT are reliable, but slower
# GUROBI is best
CONVEX_SOLVERS = ["GUROBI"]  # ["SCS", "CVXOPT", "GUROBI", "ECOS"]

# ECOS_BB & GLPK_MI are open source alternative, but much slower
MIQP_SOLVERS = ["GUROBI"]

# Set to small values bc gurobi non-commercial can not solver large model.
N_FEATURES = [20, 30]  # an overdetermined and underdetermined case
N_SAMPLES = 25
N_INFORMATIVE = 10


@pytest.fixture(scope="package")
def rng():
    """Seed and return an RNG for test reproducibility"""
    return np.random.default_rng(SEED)


@pytest.fixture(params=CONVEX_SOLVERS)
def solver(request):
    return request.param


@pytest.fixture(params=MIQP_SOLVERS)
def solver(request):
    return request.param


@pytest.fixture(scope="package", params=N_FEATURES)
def random_model(rng, request):
    """Returns a fully random set of X, y, and beta representing a linear model."""
    X, y, beta = make_regression(
        n_samples=N_SAMPLES,
        n_features=request.param,
        n_informative=N_INFORMATIVE,
        coef=True,
        random_state=rng.integers(0, 2**32 - 1),
        bias=10 * rng.random(),
    )
    return X, y, beta


@pytest.fixture(scope="package", params=N_FEATURES)
def random_energy_model(rng, request):
    """Returns a random set of X, y, and beta with added gaussian noise for a linear
    model with sparse coefficients beta decay (on average) exponentially with the index
    of the coefficient.
    """
    X = rng.random((N_SAMPLES, request.param))
    beta = np.zeros(request.param)  # coefficients
    non_zero_ids = rng.choice(request.param, size=N_INFORMATIVE, replace=False)
    non_zero_ids = np.array(np.round(non_zero_ids), dtype=int)

    for idx in non_zero_ids:
        eci = 0
        mag = np.exp(-0.5 * idx)
        while np.isclose(eci, 0):
            eci = (rng.random() - 0.5) * 2 * mag
        beta[idx] = eci
    y = X @ beta + rng.normal(size=N_SAMPLES) * 2e-3  # fake energies
    return X, y, beta


@pytest.fixture(scope="package")
def sparse_coded_signal(rng):
    n_samples, n_features, n_nonzero, n_targets = 10, 30, 5, 3
    y, X, beta = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_features,
        n_features=n_samples,
        n_nonzero_coefs=n_nonzero,
        random_state=rng.integers(0, 2**32 - 1),
        data_transposed=True,
    )
    # Make X not of norm 1 for testing
    X *= 10
    y *= 10
    return X, y[:, 0], beta[:, 0]
