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
MIQP_estimators = [
    BestSubsetSelection,
    RidgedBestSubsetSelection,
    RegularizedL0,
    L2L0,
]

THRESHOLD = 1e-12


def assert_hierarchy_respected(coef, slack_z, hierarchy, groups=None):
    groups = groups if groups is not None else np.arange(len(coef))
    group_ids = np.unique(groups)
    for grp_id, active, parents in zip(group_ids, slack_z, hierarchy):
        if active == 1:  # all parents must also be active
            assert all(
                (abs(coef[groups == parent]) >= THRESHOLD).all() for parent in parents
            )


def test_perfect_signal_recovery(sparse_coded_signal):
    X, y, beta = sparse_coded_signal
    X = X.T

    (idx,) = beta.nonzero()

    estimator = BestSubsetSelection(sparse_bound=np.count_nonzero(beta))
    estimator.fit(X, y)

    npt.assert_array_equal(idx, np.flatnonzero(estimator.coef_))
    npt.assert_array_almost_equal(beta, estimator.coef_)

    r_estimator = RidgedBestSubsetSelection(sparse_bound=np.count_nonzero(beta))

    # very low regularization should be the same
    r_estimator.eta = 1e-16
    r_estimator.fit(X, y)
    npt.assert_array_almost_equal(beta, r_estimator.coef_)
    npt.assert_array_equal(idx, np.flatnonzero(r_estimator.coef_))
    assert all(i in np.flatnonzero(r_estimator.coef_) for i in idx)

    # a bit higher regularization, check shrinkage
    coef = r_estimator.coef_.copy()
    r_estimator.eta = 1e-4
    r_estimator.fit(X, y)
    npt.assert_array_almost_equal(beta, r_estimator.coef_, decimal=1)
    assert np.linalg.norm(coef) > np.linalg.norm(r_estimator.coef_)

    # very sensitive to the value of alpha for exact results
    estimator = RegularizedL0(alpha=0.0008)
    estimator.fit(X, y)

    npt.assert_array_equal(idx, np.flatnonzero(estimator.coef_))
    npt.assert_array_almost_equal(beta, estimator.coef_, decimal=2)


@pytest.mark.parametrize("estimator_cls", MIQP_estimators)
def test_slack_variables(estimator_cls, random_model_with_groups, miqp_solver, rng):
    X, y, beta, groups = random_model_with_groups

    # ignore groups
    if "Subset" in estimator_cls.__name__:
        estimator = estimator_cls(sparse_bound=len(beta) // 2, solver=miqp_solver)
    else:
        estimator = estimator_cls(alpha=3.0, solver=miqp_solver)

    estimator.fit(X, y)
    for coef, active in zip(
        estimator.coef_, estimator.canonicals_.auxiliaries.z0.value
    ):
        if active == 1:
            assert abs(coef) >= THRESHOLD
        else:
            assert abs(coef) < THRESHOLD

    # now group hierarchy
    group_ids = np.sort(np.unique(groups))
    if "Subset" in estimator_cls.__name__:
        estimator = estimator_cls(
            groups, sparse_bound=len(group_ids) // 2, solver=miqp_solver
        )
    else:
        estimator = estimator_cls(groups, alpha=2.0, solver=miqp_solver)

    estimator.fit(X, y)
    for gid, active in zip(group_ids, estimator.canonicals_.auxiliaries.z0.value):
        if active:
            assert all(abs(estimator.coef_[groups == gid]) >= THRESHOLD)
        else:
            assert all(abs(estimator.coef_[groups == gid]) < THRESHOLD)


@pytest.mark.parametrize("estimator_cls", MIQP_estimators)
def test_singleton_hierarchy(estimator_cls, random_model, miqp_solver, rng):
    X, y, beta = random_model
    (idx,) = beta.nonzero()

    # ignore groups, single covariate hierarchy
    if "Subset" in estimator_cls.__name__:
        estimator = estimator_cls(sparse_bound=len(beta) // 2, solver=miqp_solver)
    else:
        estimator = estimator_cls(alpha=2.0, solver=miqp_solver)

    fully_chained = [[len(beta) - 1]] + [[i] for i in range(0, len(beta) - 1)]
    estimator.hierarchy = fully_chained
    estimator.fit(X, y)

    # bound is set lower than number of coefs so all must be zero in BestSubset
    if any(estimator.coef_ == 0):
        assert all(estimator.coef_ == 0)
    else:
        assert all(estimator.coef_ != 0)
    assert_hierarchy_respected(
        estimator.coef_, estimator.canonicals_.auxiliaries.z0.value, fully_chained
    )

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
    estimator.problem = None
    estimator.fit(X, y)
    assert_hierarchy_respected(
        estimator.coef_, estimator.canonicals_.auxiliaries.z0.value, hierarchy
    )


@pytest.mark.parametrize("estimator_cls", MIQP_estimators)
def test_group_hierarchy(estimator_cls, random_model_with_groups, miqp_solver, rng):
    X, y, beta, groups = random_model_with_groups
    (idx,) = beta.nonzero()

    # now group hierarchy
    group_ids = np.unique(groups)
    if "Subset" in estimator_cls.__name__:
        estimator = estimator_cls(
            groups, sparse_bound=len(group_ids) // 2, solver=miqp_solver
        )
    else:
        estimator = estimator_cls(groups, alpha=3.0, solver=miqp_solver)

    fully_chained = [[group_ids[-1]]] + [
        [group_ids[i]] for i in range(0, len(group_ids) - 1)
    ]
    estimator.hierarchy = fully_chained
    estimator.fit(X, y)

    # bound is set lower than number of coefs so all must be zero in BestSubset
    if any(estimator.coef_ == 0):
        assert all(estimator.coef_ == 0)
    else:
        assert all(estimator.coef_ != 0)

    assert_hierarchy_respected(
        estimator.coef_,
        estimator.canonicals_.auxiliaries.z0.value,
        fully_chained,
        groups=groups,
    )

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
        if 0 < i < len(group_ids) // 2 and i not in [grp1, grp2]:
            hierarchy[i].append(grp2)

    estimator.problem = None  # TODO also remove this...
    estimator.hierarchy = hierarchy
    estimator.fit(X, y)

    assert_hierarchy_respected(
        estimator.coef_,
        estimator.canonicals_.auxiliaries.z0.value,
        hierarchy,
        groups=groups,
    )


def test_set_parameters(random_model):
    X, y, beta = random_model
    estimator = RidgedBestSubsetSelection(sparse_bound=1, eta=1.0)
    estimator.sparse_bound = 2
    estimator.fit(X, y)
    assert estimator.canonicals_.parameters.sparse_bound.value == 2
    assert estimator.canonicals_.parameters.eta.value == 1.0

    estimator.eta = 0.5
    estimator.fit(X, y)
    assert estimator.canonicals_.parameters.eta.value == 0.5


def test_bad_input(random_model):
    X, y, beta = random_model

    # bad sparse_bound
    estimator = BestSubsetSelection(sparse_bound=-1)
    with pytest.raises(ValueError):
        estimator.fit(X, y)

    # bad eta
    estimator = RidgedBestSubsetSelection(eta=-1.0)
    with pytest.raises(ValueError):
        estimator.fit(X, y)
