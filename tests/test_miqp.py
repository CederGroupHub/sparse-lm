import numpy as np
import numpy.testing as npt
import pytest

from sparselm.model import BestSubsetSelection, RidgedBestSubsetSelection, RegularizedL0, L2L0, L1L0


def test_perfect_signal_recovery(sparse_coded_signal):
    X, y, beta = sparse_coded_signal
    (idx,) = beta.nonzero()

    estimator = BestSubsetSelection(
        groups=np.arange(len(beta)), sparse_bound=np.count_nonzero(beta)
    )
    estimator.fit(X, y)

    npt.assert_array_equal(idx, np.flatnonzero(estimator.coef_))
    npt.assert_array_almost_equal(beta, estimator.coef_)

    r_estimator = RidgedBestSubsetSelection(
        groups=np.arange(len(beta)), sparse_bound=np.count_nonzero(beta)
    )

    # very low regularization should be the same
    r_estimator.eta = 1e-10
    r_estimator.fit(X, y)
    npt.assert_array_almost_equal(beta, r_estimator.coef_, decimal=2)
    assert all(i in np.flatnonzero(r_estimator.coef_) for i in idx)

    # a bit higher regularization, check shrinkage
    coef = estimator.coef_.copy()
    r_estimator.eta = 1e-4
    r_estimator.fit(X, y)
    npt.assert_array_almost_equal(beta, r_estimator.coef_, decimal=2)
    assert all(i in np.flatnonzero(r_estimator.coef_) for i in idx)
    assert np.linalg.norm(coef) > np.linalg.norm(r_estimator.coef_)

    # very sensitive to the value of alpha for exact results
    estimator = RegularizedL0(
        groups=np.arange(len(beta)), alpha=0.025
    )
    estimator.fit(X, y)

    npt.assert_array_equal(idx, np.flatnonzero(estimator.coef_))
    npt.assert_array_almost_equal(beta, estimator.coef_)


def test_set_parameters():
    estimator = RidgedBestSubsetSelection(groups=[0, 1, 2], sparse_bound=1)
    estimator.sparse_bound = 2
    assert estimator.sparse_bound == 2
    assert estimator._bound.value == 2

    estimator.eta = 0.5
    assert estimator.eta == 0.5
    assert estimator._eta.value == 0.5


def test_bad_input():
    with pytest.raises(ValueError):
        estimator = BestSubsetSelection(groups=[0, 1, 2], sparse_bound=-1)

    estimator = BestSubsetSelection(groups=[0, 1, 2], sparse_bound=1)
    with pytest.raises(ValueError):
        estimator.sparse_bound = 0
