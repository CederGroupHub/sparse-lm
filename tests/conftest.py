import numpy as np
import pytest
from sklearn.datasets import make_regression, make_sparse_coded_signal

SEED = 0

# A few solvers to test for convex problems
# ECOS sometimes fails for Adaptive group estimators, but is fast
# SCS and CXVOPT are reliable, but slower
# GUROBI is best
CONVEX_SOLVERS = ["GUROBI", "ECOS"]  # SCS, GUROBI, CVXOPT

# ECOS_BB is open source alternative, but much slower, and can get things wrong
MIQP_SOLVERS = ["GUROBI"]  # SCIP fails some tests...

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
def miqp_solver(request):
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
    n_components, n_features, n_nonzero = 24, 12, 6
    y, X, beta = make_sparse_coded_signal(
        n_samples=1,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero,
        random_state=rng.integers(0, 2**32 - 1),
    )
    return X, y, beta


@pytest.fixture(params=[4, 6], scope="package")
def random_model_with_groups(random_model, rng, request):
    """Add a correct set of groups to model."""
    X, y, beta = random_model
    n_groups = request.param
    n_active_groups = n_groups // 3 + 1

    n_features_per_group = len(beta) // n_groups
    active_group_inds = rng.choice(range(n_groups), size=n_active_groups, replace=False)
    inactive_group_inds = np.setdiff1d(range(n_groups), active_group_inds)

    groups = np.zeros(len(beta), dtype=int)
    active_feature_inds = np.where(abs(beta) > 0)[0]
    inactive_feature_inds = np.setdiff1d(np.arange(len(beta)), active_feature_inds)

    # set active groups
    for i in active_group_inds:
        if len(active_feature_inds) > n_features_per_group:
            group_inds = rng.choice(
                active_feature_inds, size=n_features_per_group, replace=False
            )
        else:
            group_inds = active_feature_inds
        groups[group_inds] = i
        active_feature_inds = np.setdiff1d(active_feature_inds, group_inds)

    # set inactive_groups
    for i in inactive_group_inds:
        if len(inactive_feature_inds) > n_features_per_group:
            group_inds = rng.choice(
                inactive_feature_inds, size=n_features_per_group, replace=False
            )
        else:
            group_inds = inactive_feature_inds
        groups[group_inds] = i
        inactive_feature_inds = np.setdiff1d(inactive_feature_inds, group_inds)

    return X, y, beta, groups
