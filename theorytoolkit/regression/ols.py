"""Ordinary least squares cvxpy solver."""

__author__ = "Fengyu Xie, Luis Barroso-Luque"

import cvxpy as cp
from .base import CVXEstimator


class OrdinaryLeastSquares(CVXEstimator):
    """
    OLS Linear Regression Estimator implemented with cvxpy.
    """

    def _initialize_problem(self, X, y):
        super()._initialize_problem(X, y)
        objective = cp.sum_squares(X @ self._beta - y)
        self._problem = cp.Problem(cp.Minimize(objective))
