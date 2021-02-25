"""Ordinary least squares solver."""

__author__ = "Fengyu Xie"

import numpy as np
import warnings
import math
from .base import BaseEstimator


class OLSEstimator(BaseEstimator):
    """
    Lasso Estimator implemented with cvxpy.
    """

    def __init__(self):
        super().__init__()

    def fit(self, feature_matrix, target_vector, sample_weight=None,**kwargs):
        """
        Fit the estimator. If mu not given, will optimize it.
        Inputs:
            feature_matrix(2d ArrayLike, n_structures*n_bit_orbits):
                Feature matrix of structures.
            target_vector(1d ArrayLike):
                Physical properties to fit.
            sample_weight(1d ArrayLike or None):
                Weight of samples. If not given, rows will be treated with equal weights.
        Return:
            No return value.
            Fitter coefficients storeed in self.coef_.
        """
        #No mu needs to be optimized.

        super().fit(feature_matrix, target_vector,
                    sample_weight=sample_weight)

    def _solve(self, feature_matrix, target_vector):
        """
        X and y should already have been adjusted to account for weighting.
        """
        A = feature_matrix.copy()
        b = target_vector.copy()

        return np.linalg.inv(A.T@A)@A.T@b
