"""Ordinary least squares cvxpy solver."""

__author__ = "Fengyu Xie, Luis Barroso-Luque"


from types import SimpleNamespace
from typing import Optional

import cvxpy as cp
from numpy.typing import ArrayLike

from ._base import CVXRegressor


class OrdinaryLeastSquares(CVXRegressor):
    """OLS Linear Regression Estimator implemented with cvxpy."""

    def _generate_objective(
        self,
        X: ArrayLike,
        y: ArrayLike,
        beta: cp.Variable,
        parameters: Optional[SimpleNamespace] = None,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        return 1 / (2 * X.shape[0]) * cp.sum_squares(X @ beta - y)
