import numpy as np
import numpy.testing as npt
import pytest

from sparselm.dataset import make_group_regression


@pytest.mark.parametrize("n_informative_groups", [5, 20])
@pytest.mark.parametrize("n_features_per_group", [5, 4 * list(range(2, 7))])
@pytest.mark.parametrize("frac_informative_in_group", [1.0, 0.5])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("coef", [True, False])
def test_make_group_regression(
    n_informative_groups, n_features_per_group, frac_informative_in_group, shuffle, coef
):
    model = make_group_regression(
        n_informative_groups=n_informative_groups,
        n_features_per_group=n_features_per_group,
        frac_informative_in_group=frac_informative_in_group,
        shuffle=shuffle,
        coef=coef,
    )

    assert len(model) == 4 if coef else 3

    if coef:
        X, y, groups, coefs = model
    else:
        X, y, groups = model

    if not isinstance(n_features_per_group, list):
        n_features_per_group = [n_features_per_group] * 20

    n_features = (
        sum(n_features_per_group)
        if isinstance(n_features_per_group, list)
        else 20 * n_features_per_group
    )

    assert X.shape == (100, n_features)
    assert y.shape == (100,)
    assert groups.shape == (n_features,)
    assert len(np.unique(groups)) == 20

    if coef:
        n_informative = sum(
            round(frac_informative_in_group * n_features_per_group[i])
            for i in range(n_informative_groups)
        )

        assert coefs.shape == (n_features,)
        assert sum(coef > 0 for coef in coefs) == n_informative
        npt.assert_array_almost_equal(np.dot(X, coefs), y)

    if shuffle:
        # check that not all groups are lumped together
        assert sum(np.diff(groups) == 0) < 20 - 1

    # check warning
    with pytest.warns(UserWarning):
        make_group_regression(frac_informative_in_group=1 / 100)
