"""Ordinary least squares cvxpy solver."""

__author__ = "Fengyu Xie, Luis Barroso-Luque"

import cvxpy as cp

from ._base import CVXEstimator


class OrdinaryLeastSquares(CVXEstimator):
    """OLS Linear Regression Estimator implemented with cvxpy."""

    def _generate_objective(self, X, y, beta, parameters=None):
        return 1 / (2 * X.shape[0]) * cp.sum_squares(X @ beta - y)
