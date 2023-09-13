"""Ordinary least squares cvxpy solver."""

from __future__ import annotations

__author__ = "Fengyu Xie, Luis Barroso-Luque"


from types import SimpleNamespace

import cvxpy as cp
from numpy.typing import NDArray

from ._base import CVXRegressor


class OrdinaryLeastSquares(CVXRegressor):
    r"""Ordinary Least Squares Linear Regression.

    Regression objective:

    .. math::

        \min_{\beta} || X \beta - y ||^2_2

    Args:
        fit_intercept (bool):
            Whether the intercept should be estimated or not.
            If False, the data is assumed to be already centered.
        copy_X (bool):
            If True, X will be copied; else, it may be overwritten.
        warm_start (bool):
            When set to True, reuse the solution of the previous call to
            fit as initialization, otherwise, just erase the previous
            solution.
        solver (str):
            cvxpy backend solver to use. Supported solvers are listed here:
            https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
        solver_options (dict):
            dictionary of keyword arguments passed to cvxpy solve.
            See docs linked above for more information.

    Attributes:
        coef_ (NDArray):
            Parameter vector (:math:`\beta` in the cost function formula) of shape (n_features,).
        intercept_ (float):
            Independent term in decision function.
        canonicals_ (SimpleNamespace):
            Namespace that contains underlying cvxpy objects used to define
            the optimization problem. The objects included are the following:
                - objective - the objective function.
                - beta - variable to be optimized (corresponds to the estimated coef_ attribute).
                - parameters - hyper-parameters
                - auxiliaries - auxiliary variables and expressions
                - constraints - solution constraints
    """

    def _generate_objective(
        self,
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
    ) -> cp.Expression:
        return 1 / (2 * X.shape[0]) * cp.sum_squares(X @ beta - y)
