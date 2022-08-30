import numpy as np
import pytest

# Set to small values bc gurobi non-commercial can not solver large model.
n_features = 30
n_samples = 40
n_nonzeros = 20


@pytest.fixture(scope="package")
def random_model():
    femat = np.random.rand(n_samples, n_features)
    ecis = np.zeros(n_features)
    non_zero_ids = np.random.choice(n_features, size=n_nonzeros, replace=False)
    non_zero_ids = np.array(np.round(non_zero_ids), dtype=int)
    for idx in non_zero_ids:
        eci = 0
        mag = np.exp(-0.5 * idx)
        while np.isclose(eci, 0):
            eci = (np.random.random() - 0.5) * 2 * mag
        ecis[idx] = eci
    energies = femat @ ecis + np.random.normal(size=n_samples) * 2e-3
    return femat, energies, ecis
