import numpy as np
import pytest
from sklearn.datasets import make_regression

SEED = None

# Set to small values bc gurobi non-commercial can not solver large model.
N_FEATURES = [20, 30]  # an overdetermined and underdetermined case
N_SAMPLES = 25
N_INFORMATIVE = 10


@pytest.fixture(scope="package")
def rng():
    """Seed and return an RNG for test reproducibility"""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="package", params=N_FEATURES)
def random_model(rng, request):
    """Returns a fully random set of X, y, and beta representing a linear model."""
    X, y, beta = make_regression(
        n_samples=N_SAMPLES,
        n_features=request.param,
        n_informative=N_INFORMATIVE,
        coef=True,
        random_state=rng.integers(0, 2**32 - 1),
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
