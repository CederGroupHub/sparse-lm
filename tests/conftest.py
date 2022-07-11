import pytest
import numpy as np

n_features = 150
n_samples = 100
n_nonzeros = 80


@pytest.fixture(scope="package")
def random_model():
    femat = np.random.random((n_samples, n_features))
    ecis = np.zeros(n_features)
    non_zero_ids = np.random.sample(n_features, size=n_nonzeros, replace=False)
    non_zero_ids = np.array(np.round(non_zero_ids), dtype=int)
    for idx in non_zero_ids:
        eci = 0
        mag = np.exp(-0.1 * idx)
        while np.isclose(eci, 0):
            eci = (np.random.random() - 0.5) * mag
        ecis[idx] = eci
    energies = femat @ ecis + np.random.normal(size=n_samples) * 2E-3
    return femat, energies, ecis

