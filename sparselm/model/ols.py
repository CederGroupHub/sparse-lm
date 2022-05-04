"""Ordinary least squares cvxpy solver."""

__author__ = "Fengyu Xie, Luis Barroso-Luque"

import cvxpy as cp
from .base import CVXEstimator


class OrdinaryLeastSquares(CVXEstimator):
    """
    OLS Linear Regression Estimator implemented with cvxpy.
    """

    def _gen_objective(self, X, y):
        return 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y)
