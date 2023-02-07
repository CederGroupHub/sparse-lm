import numpy as np
import numpy.testing as npt
import pytest

from sparselm.model import (
    L2L0,
    BestSubsetSelection,
    RegularizedL0,
    RidgedBestSubsetSelection,
)

# exclude L1L0 since it breaks hierarchy constraints...
MIQP_estimators = [BestSubsetSelection, RidgedBestSubsetSelection, RegularizedL0, L2L0]

THRESHOLD = 1e-4


def assert_hierarchy_respected(coef, hierarchy, groups=None):
    groups = groups if groups is not None else np.arange(len(coef))
    for high_id, sub_ids in enumerate(hierarchy):
        for sub_id in sub_ids:
            """
            if abs(coef[groups == groups[sub_id]][0]) < THRESHOLD:
                print("------INACTIVE COEF CONSTRAINT--------------")
                print(sub_id, high_id)
                print(abs(coef[groups == groups[sub_id]]))
                print(abs(coef[groups == groups[high_id]]))
                assert all(abs(coef[groups == groups[sub_id]]) < THRESHOLD)
                assert all(abs(coef[groups == groups[high_id]]) < THRESHOLD)
            else:
                # print("always on!")
                assert all(abs(coef[groups == groups[sub_id]]) > THRESHOLD)
            """
            if coef[groups == groups[sub_id]][0] == 0:
                print("------INACTIVE COEF CONSTRAINT--------------")
                print(sub_id, high_id)
                print(abs(coef[groups == groups[sub_id]]))
                print(abs(coef[groups == groups[high_id]]))
                assert all(coef[groups == groups[sub_id]] == 0)
                assert all(coef[groups == groups[high_id]] == 0)
            else:
                # print("always on!")
                assert all(coef[groups == groups[sub_id]] != 0)


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
    npt.assert_array_almost_equal(beta, r_estimator.coef_)
    assert all(i in np.flatnonzero(r_estimator.coef_) for i in idx)

    # a bit higher regularization, check shrinkage
    coef = estimator.coef_.copy()
    r_estimator.eta = 1e-4
    r_estimator.fit(X, y)
    npt.assert_array_almost_equal(beta, r_estimator.coef_, decimal=4)
    assert all(i in np.flatnonzero(r_estimator.coef_) for i in idx)
    assert np.linalg.norm(coef) > np.linalg.norm(r_estimator.coef_)

    # very sensitive to the value of alpha for exact results
    estimator = RegularizedL0(groups=np.arange(len(beta)), alpha=0.03)
    estimator.fit(X, y)

    npt.assert_array_equal(idx, np.flatnonzero(estimator.coef_))
    npt.assert_array_almost_equal(beta, estimator.coef_, decimal=4)


@pytest.mark.parametrize("estimator_cls", MIQP_estimators)
def test_singleton_hierarchy(estimator_cls, random_model, solver, rng):
    X, y, beta = random_model
    (idx,) = beta.nonzero()

    # ignore groups, single covariate hierarchy
    no_groups = np.arange(len(beta))
    if hasattr(estimator_cls, "sparse_bound"):
        estimator = estimator_cls(no_groups, sparse_bound=len(beta) // 2, solver=solver)
    else:
        estimator = estimator_cls(no_groups, alpha=2.0, solver=solver)

    fully_chained = [[len(beta) - 1]] + [[i] for i in range(0, len(beta) - 1)]
    estimator.hierarchy = fully_chained
    estimator.fit(X, y)

    # bound is set lower than number of coefs so all must be zero in BestSubset
    if any(estimator.coef_ == 0):
        assert all(estimator.coef_ == 0)
    else:
        assert all(estimator.coef_ != 0)

    hierarchy = []
    for i in range(len(beta)):
        # everything depends on 1st nonzero coef
        if i != idx[0]:
            hierarchy.append([idx[0]])
        else:
            hierarchy.append([])
        # first half of remaining depends on 2nd nonzero
        if 0 < i < len(beta) // 2 and i != idx[1]:
            hierarchy[i].append(idx[1])
        # second half of remaining on 3rd nonzero
        if len(beta) // 2 <= i and i != idx[2]:
            hierarchy[i].append(idx[2])

    estimator.hierarchy = hierarchy
    # TODO make hierarchy and other non cp.Parameter params reset problem if reset
    estimator._problem = None
    estimator.fit(X, y)
    assert_hierarchy_respected(estimator.coef_, hierarchy)


@pytest.mark.parametrize("estimator_cls", MIQP_estimators)
def test_group_hierarchy(estimator_cls, random_model_with_groups, solver, rng):
    X, y, beta, groups = random_model_with_groups
    (idx,) = beta.nonzero()

    # now group hierarchy
    group_ids = np.unique(groups)
    if hasattr(estimator_cls, "sparse_bound"):
        estimator = estimator_cls(
            groups, sparse_bound=len(group_ids) // 2, solver=solver
        )
    else:
        estimator = estimator_cls(groups, alpha=3.0, solver=solver)

    fully_chained = [[len(group_ids) - 1]] + [[i] for i in range(0, len(group_ids) - 1)]
    estimator.hierarchy = fully_chained
    estimator.fit(X, y)

    # bound is set lower than number of coefs so all must be zero in BestSubset
    if any(estimator.coef_ == 0):
        assert all(estimator.coef_ == 0)
    else:
        assert all(estimator.coef_ != 0)

    # pick two groups with nozero coefs
    grp1 = groups[idx[0]]
    while (grp2 := groups[rng.choice(idx)]) == grp1:
        pass

    hierarchy = []
    for i in range(len(group_ids)):
        # everything depends on 1st nonzero coef
        if i != grp1:
            hierarchy.append([grp1])
        else:
            hierarchy.append([])
        # first half of remaining depends on 2nd nonzero
        if 0 < i < len(group_ids) // 2 and i != grp2:
            hierarchy[i].append(grp2)

    estimator._problem = None  # TODO also remove this...
    estimator.hierarchy = hierarchy
    estimator.fit(X, y)
    print(estimator._z0.value)
    print(hierarchy)
    assert_hierarchy_respected(estimator.coef_, hierarchy, groups)


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
