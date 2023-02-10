"""Data and hyper-parameters validation utilities."""

import numpy as np


def _check_groups(groups, n_features):
    """Check that groups are 1D and of the correct length.

    Args:
        groups (list or ndarray):
            List of group labels
        n_features (int):
            Number of features/covariates being fit

    Returns:
        ndarray: groups as a 1D ndarray
    """
    if groups is None:
        groups = np.arange(n_features)

    if not isinstance(groups, (list, np.ndarray)):
        raise TypeError("groups must be a list or ndarray")

    groups = np.asarray(groups).astype(int)
    if groups.ndim != 1:
        raise ValueError("groups must be a 1D array")

    if len(groups) != n_features:
        raise ValueError(
            f"groups must be the same length as the number of features {n_features}"
        )
    return groups


def _check_group_weights(group_weights, groups):
    """Check that group weights are 1D and of the correct length.

    Args:
        group_weights (list or ndarray):
            List of group weights
        groups (list or ndarray):
            List of group labels

    Returns:
        ndarray: group weights as a 1D ndarray
    """
    group_ids = np.sort(np.unique(groups))

    if group_weights is None:
        group_weights = np.sqrt([sum(groups == i) for i in group_ids])

    if not isinstance(group_weights, (list, np.ndarray)):
        raise TypeError("group_weights must be a list or ndarray")

    group_weights = np.asarray(group_weights)
    if len(group_weights) != len(group_ids):
        raise ValueError(
            f"group_weights must be the same length as the number of groups {len(group_weights)} != {len(group_ids)}"
        )
    return group_weights
