"""MIQP based solvers for Best Subset Selection solutions.

Allows hierarchy constraints similar to mixed L0 solvers.
"""

from __future__ import annotations

__author__ = "Luis Barroso-Luque"

from numbers import Real
from types import SimpleNamespace
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval

from sparselm.model._base import TikhonovMixin

from ._base import MIQPl0


class BestSubsetSelection(MIQPl0):
    r"""MIQP Best Subset Selection Regressor.

    Generalized best subset that allows grouping subsets.

    Args:
        groups (NDArray):
            array-like of integers specifying groups. Length should be the
            same as model, where each integer entry specifies the group
            each parameter corresponds to. If no grouping is required,
            simply pass a list of all different numbers, i.e. using range.
        sparse_bound (int):
            Upper bound on sparsity. The upper bound on total number of
            nonzero coefficients.
        big_M (float):
            Upper bound on the norm of coefficients associated with each
            groups of coefficients :math:`||\beta_c||_2`.
        hierarchy (list):
            A list of lists of integers storing hierarchy relations between
            coefficients.
            Each sublist contains indices of other coefficients
            on which the coefficient associated with each element of
            the list depends. i.e. hierarchy = [[1, 2], [0], []] mean that
            coefficient 0 depends on 1, and 2; 1 depends on 0, and 2 has no
            dependence.
        ignore_psd_check (bool):
            Whether to ignore cvxpy's PSD checks of matrix used in
            quadratic form. Default is True to avoid raising errors for
            poorly conditioned matrices. But if you want to be strict set
            to False.
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
            See docs in CVXRegressor for more information.

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

    Notes:
        Installation of Gurobi is not a must, but highly recommended. An open source alternative
        is SCIP. ECOS_BB also works but can be very slow, and has recurring correctness issues.
        See the Mixed-integer programs section of the cvxpy docs:
        https://www.cvxpy.org/tutorial/advanced/index.html

    WARNING:
        Even with gurobi solver, this can take a very long time to converge for large problems and under-determined
        problems.
    """

    _cvx_parameter_constraints: dict[str, list[Any]] = {
        "sparse_bound": [Interval(type=Real, left=0, right=None, closed="left")],
        **MIQPl0._cvx_parameter_constraints,
    }

    def __init__(
        self,
        groups: NDArray | None = None,
        sparse_bound=100,
        big_M: int = 100,
        hierarchy: list[list[int]] | None = None,
        ignore_psd_check: bool = True,
        fit_intercept: bool = False,
        copy_X: bool = True,
        warm_start: bool = False,
        solver: str | None = None,
        solver_options: dict | None = None,
    ):
        super().__init__(
            groups=groups,
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )
        self.sparse_bound = sparse_bound

    def _generate_constraints(
        self,
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
    ) -> list[cp.Constraint]:
        """Generate the constraints for best subset selection."""
        constraints = super()._generate_constraints(X, y, beta, parameters, auxiliaries)
        constraints += [cp.sum(auxiliaries.z0) <= parameters.sparse_bound]  # type: ignore
        return constraints


class RidgedBestSubsetSelection(TikhonovMixin, BestSubsetSelection):
    r"""MIQP best subset selection Regressor with Ridge/Tihkonov regularization.

    Args:
        groups (NDArray):
            array-like of integers specifying groups. Length should be the
            same as model, where each integer entry specifies the group
            each parameter corresponds to. If no grouping is required,
            simply pass a list of all different numbers, i.e. using range.
        sparse_bound (int):
            Upper bound on sparsity. The upper bound on total number of
            nonzero coefficients.
        eta (float):
            L2 regularization hyper-parameter.
        big_M (float):
            Upper bound on the norm of coefficients associated with each
            groups of coefficients :math:`||\beta_c||_2`.
        hierarchy (list):
            A list of lists of integers storing hierarchy relations between
            coefficients.
            Each sublist contains indices of other coefficients
            on which the coefficient associated with each element of
            the list depends. i.e. hierarchy = [[1, 2], [0], []] mean that
            coefficient 0 depends on 1, and 2; 1 depends on 0, and 2 has no
            dependence.
        tikhonov_w (np.array):
            Matrix to add weights to L2 regularization.
        ignore_psd_check (bool):
            Whether to ignore cvxpy's PSD checks of matrix used in
            quadratic form. Default is True to avoid raising errors for
            poorly conditioned matrices. But if you want to be strict set
            to False.
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
            See docs in CVXRegressor for more information.

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

    Notes:
        Installation of Gurobi is not a must, but highly recommended. An open source alternative
        is SCIP. ECOS_BB also works but can be very slow, and has recurring correctness issues.
        See the Mixed-integer programs section of the cvxpy docs:
        https://www.cvxpy.org/tutorial/advanced/index.html

    WARNING:
        Even with gurobi solver, this can take a very long time to converge for large problems and under-determined
        problems.
    """

    _cvx_parameter_constraints: dict[str, list[Any]] = {
        "eta": [Interval(type=Real, left=0.0, right=None, closed="left")],
        **BestSubsetSelection._cvx_parameter_constraints,
    }

    def __init__(
        self,
        groups: NDArray | None = None,
        sparse_bound: int = 100,
        eta: float = 1.0,
        big_M: int = 100,
        hierarchy: list[list[int]] | None = None,
        tikhonov_w: NDArray[np.floating] | None = None,
        ignore_psd_check: bool = True,
        fit_intercept: bool = False,
        copy_X: bool = True,
        warm_start: bool = False,
        solver: str | None = None,
        solver_options: dict | None = None,
    ):
        super().__init__(
            groups=groups,
            sparse_bound=sparse_bound,
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )
        self.tikhonov_w = tikhonov_w
        self.eta = eta
