"""Generate synthemetic datasets akin to sklearn.datasets."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from numpy.random import RandomState
from sklearn.datasets import make_regression
from sklearn.utils import check_random_state


def make_group_regression(
    n_samples: int = 100,
    n_groups: int = 20,
    n_features_per_group: int | Sequence = 10,
    n_informative_groups: int = 5,
    frac_informative_in_group: float = 1.0,
    bias: float = 0.0,
    effective_rank: int | None = None,
    tail_strength: float = 0.5,
    noise: float = 0.0,
    shuffle: bool = True,
    coef: bool = False,
    random_state: int | RandomState | None = None,
) -> tuple[np.ndarray, ...]:
    """Generate a random regression problem with grouped covariates.

    Args:
        n_samples (int, optional):
            Number of samples to generate.
        n_groups (int, optional):
            Number of groups to generate.
        n_features_per_group (int | Sequence, optional):
            Number of features per group to generate. If a sequence is passed the
            length must be equal to n_groups then each element will be the number of
            features in the corresponding group.
        n_informative_groups (int, optional):
            Number of informative groups.
        frac_informative_in_group (float, optional):
            Fraction of informative features in each group
            The number of features will be rounded to nearest int.
        bias (float, optional):
            Bias added to the decision function.
        effective_rank ([type], optional):
            Approximate number of singular vectors
            required to explain most of the input data by linear combinations.
        tail_strength (float, optional):
            Relative importance of the fat noisy tail
            of the singular values profile if `effective_rank` is not None.
        noise (float, optional):
            Standard deviation of the gaussian noise applied to the output.
        shuffle (bool, optional):
            Shuffle the samples and the features. Defaults to True.
        coef (bool, optional):
            If True, the coefficients of the underlying linear model are returned.
        random_state ([type], optional):
            Random state for dataset generation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, ...]:
            X, y, groups, coefficients (optional)
    """
    generator = check_random_state(random_state)

    informative_groups = list(range(n_informative_groups))

    if isinstance(n_features_per_group, int):
        n_features = n_features_per_group * n_groups
        n_informative_in_group = round(frac_informative_in_group * n_features_per_group)
        n_informative = n_informative_in_group * n_informative_groups
        # make n_features_per_group a list of length n_groups
        n_features_per_group = [n_features_per_group] * n_groups
        n_informative_per_group = [n_informative_in_group] * n_informative_groups
    else:
        if len(n_features_per_group) == n_groups:
            n_features = sum(n_features_per_group)
            n_informative_per_group = [
                round(frac_informative_in_group * n_features_per_group[i])
                for i in informative_groups
            ]
            n_informative = sum(n_informative_per_group)
        else:
            raise ValueError(
                "If passing a sequence of n_features_per_group, the length must be "
                "equal to n_groups."
            )

    if any(n < 1 for n in n_informative_per_group):
        warnings.warn(
            "The number of features and fraction of informative features per group resulted in "
            "informative groups having no informative features.",
            UserWarning,
        )

    X, y, coefs = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        bias=bias,
        effective_rank=effective_rank,
        tail_strength=tail_strength,
        noise=noise,
        shuffle=shuffle,
        coef=True,
        random_state=generator,
    )

    # assign coefficients to groups
    groups = np.zeros(n_features, dtype=int)
    informative_coef_inds = np.nonzero(coefs > noise)[0].tolist()
    other_coef_inds = np.nonzero(coefs <= noise)[0].tolist()

    for i, nfg in enumerate(n_features_per_group):
        if i in informative_groups:
            nifg = n_informative_per_group[informative_groups.index(i)]
            ii = informative_coef_inds[:nifg] + other_coef_inds[: nfg - nifg]
            # remove assigned indices
            informative_coef_inds = informative_coef_inds[nifg:]
            other_coef_inds = other_coef_inds[nfg - nifg :]
        else:
            ii = other_coef_inds[:nfg]
            other_coef_inds = other_coef_inds[nfg:]

        # assign group ids
        groups[ii] = i

    if shuffle:
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        groups = groups[indices]
        coefs = coefs[indices]

    if coef:
        return X, y, groups, coefs
    else:
        return X, y, groups
