import numpy as np
import pytest
from sklearn.datasets import make_regression

SEED = None
# Set to small values bc gurobi non-commercial can not solver large model.
n_features = 30
n_samples = 40
n_informative = 20


@pytest.fixture(scope="package")
def rng():
    """Seed and return an RNG for test reproducibility"""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="package")
def random_model(rng):
    """Returns a fully random set of X, y, and beta representing a linear model."""
    X, y, beta = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        coef=True,
        random_state=rng.integers(0, 2**32 - 1),
    )
    return X, y, beta


@pytest.fixture(scope="package")
def random_energy_model(rng):
    """Returns a random set of X, y, and beta with added gaussian noise for a linear
    model with sparse coefficients beta decay (on average) exponentially with the index
    of the coefficient.
    """
    X = rng.random((n_samples, n_features))
    beta = np.zeros(n_features)  # coefficients
    non_zero_ids = rng.choice(n_features, size=n_informative, replace=False)
    non_zero_ids = np.array(np.round(non_zero_ids), dtype=int)
    for idx in non_zero_ids:
        eci = 0
        mag = np.exp(-0.5 * idx)
        while np.isclose(eci, 0):
            eci = (rng.random() - 0.5) * 2 * mag
        beta[idx] = eci
    y = X @ beta + rng.normal(size=n_samples) * 2e-3  # fake energies
    return X, y, beta


@pytest.fixture(scope="package")
def random_weights(rng):
    weights = 1000 * rng.random(n_features)
    return np.diag(weights)
