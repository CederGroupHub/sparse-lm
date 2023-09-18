"""Data and hyper-parameters validation utilities."""

import numpy as np
from numpy.typing import NDArray


def _check_groups(groups: NDArray | None, n_features: int) -> None:
    """Check that groups are 1D and of the correct length.

    Args:
        groups (NDArray):
            List of group labels
        n_features (int):
            Number of features/covariates being fit

    """
    if groups is None:
        return

    if not isinstance(groups, (list, np.ndarray)):
        raise TypeError("groups must be a list or ndarray")

    groups = np.asarray(groups).astype(int)
    if groups.ndim != 1:
        raise ValueError("groups must be a 1D array")

    if len(groups) != n_features:
        raise ValueError(
            f"groups must be the same length as the number of features {n_features}"
        )


def _check_group_weights(group_weights: NDArray | None, n_groups: int) -> None:
    """Check that group weights are 1D and of the correct length.

    Args:
        group_weights (NDArray):
            List of group weights
        n_groups (int):
            Number of groups
    """
    if group_weights is None:
        return

    if not isinstance(group_weights, (list, np.ndarray)):
        raise TypeError("group_weights must be a list or ndarray")

    group_weights = np.asarray(group_weights)
    if len(group_weights) != n_groups:
        raise ValueError(
            f"group_weights must be the same length as the number of groups {len(group_weights)} != {n_groups}"
        )
