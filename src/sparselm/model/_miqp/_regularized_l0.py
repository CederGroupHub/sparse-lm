"""MIQP based solvers for sparse solutions with hierarchical constraints.

Generalized regularized l0 solvers that allow grouping parameters as detailed in:

    https://doi.org/10.1287/opre.2015.1436

L1L0 proposed by Wenxuan Huang:

    https://arxiv.org/abs/1807.10753

L2L0 proposed by Peichen Zhong:

    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.024203

Regressors allow optional inclusion of hierarchical constraints at the single coefficient
or group of coefficients level.
"""

from __future__ import annotations

__author__ = "Luis Barroso-Luque, Fengyu Xie"


from abc import ABCMeta, abstractmethod
from numbers import Real
from types import SimpleNamespace
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval

from sparselm.model._base import TikhonovMixin

from ._base import MIQPl0


class RegularizedL0(MIQPl0):
    r"""Implementation of mixed-integer quadratic programming l0 regularized Regressor.

    Supports grouping parameters and group-level hierarchy, but requires groups as a
    compulsory argument.

    Regularized regression objective:

    .. math::

        \min_{\beta} || X \beta - y ||^2_2 + \alpha \sum_{G} z_G

    Where G represents groups of features/coefficients and :math:`z_G` is are boolean
    valued slack variables.

    Args:
        groups (NDArray):
            1D array-like of integers specifying groups. Length should be the
            same as model, where each integer entry specifies the group
            each parameter corresponds to. If no grouping is needed pass a list
            of all distinct numbers (ie range(len(coefs)) to create singleton groups
            for each parameter.
        alpha (float):
            L0 pseudo-norm regularization hyper-parameter.
        big_M (float):
            Upper bound on the norm of coefficients associated with each
            groups of coefficients :math:`||\beta_c||_2`.
        hierarchy (list):
            A list of lists of integers storing hierarchy relations between
            groups.
            Each sublist contains indices of other groups
            on which the group associated with each element of
            the list depends. i.e. hierarchy = [[1, 2], [0], []] mean that
            group 0 depends on 1, and 2; 1 depends on 0, and 2 has no
            dependence.
        ignore_psd_check (bool):
            Whether to ignore cvxpy's PSD checks  of matrix used in quadratic
            form. Default is True to avoid raising errors for poorly
            conditioned matrices. But if you want to be strict set to False.
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
    """

    _cvx_parameter_constraints: dict[str, list[Any]] = {
        "alpha": [Interval(type=Real, left=0.0, right=None, closed="left")],
        **MIQPl0._cvx_parameter_constraints,
    }

    def __init__(
        self,
        groups: NDArray | None = None,
        alpha: float = 1.0,
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
        self.alpha = alpha

    def _generate_objective(
        self,
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
    ) -> cp.Expression:
        """Generate the quadratic form and l0 regularization portion of objective."""
        c0 = 2 * X.shape[0]  # keeps hyperparameter scale independent
        objective = super()._generate_objective(
            X, y, beta, parameters, auxiliaries
        ) + c0 * parameters.alpha * cp.sum(auxiliaries.z0)
        return objective


class MixedL0(RegularizedL0, metaclass=ABCMeta):
    """Abstract base class for mixed L0 regularization models: L1L0 and L2L0."""

    _cvx_parameter_constraints: dict[str, list[Any]] = {
        "eta": [Interval(type=Real, left=0.0, right=None, closed="left")],
        **RegularizedL0._cvx_parameter_constraints,
    }

    def __init__(
        self,
        groups: NDArray | None = None,
        alpha: float = 1.0,
        eta: float = 1.0,
        big_M: int = 100,
        hierarchy: list[list[int]] | None = None,
        ignore_psd_check: bool = True,
        fit_intercept: bool = False,
        copy_X: bool = True,
        warm_start: bool = False,
        solver: str | None = None,
        solver_options: dict | None = None,
    ):
        """Initialize Regressor.

        Args:
            groups (NDArray):
                1D array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to. If no grouping is needed pass a list
                of all distinct numbers (ie range(len(coefs)) to create singleton groups
                for each parameter.
            alpha (float):
                L0 pseudo-norm regularization hyper-parameter.
            eta (float):
                standard norm regularization hyper-parameter (usually l1 or l2).
            big_M (float):
                Upper bound on the norm of coefficients associated with each

            hierarchy (list):
                A list of lists of integers storing hierarchy relations between
                coefficients.
                Each sublist contains indices of other coefficients
                on which the coefficient associated with each element of
                the list depends. i.e. hierarchy = [[1, 2], [0], []] mean that
                coefficient 0 depends on 1, and 2; 1 depends on 0, and 2 has no
                dependence.
            ignore_psd_check (bool):
                Whether to ignore cvxpy's PSD checks  of matrix used in quadratic
                form. Default is True to avoid raising errors for poorly
                conditioned matrices. But if you want to be strict set to False.
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
        """
        super().__init__(
            groups=groups,
            alpha=alpha,
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )
        self.eta = eta

    @abstractmethod
    def _generate_objective(
        self,
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
    ) -> cp.Expression:
        """Generate optimization objective."""
        # implement in derived classes using super to call MIQP_L0 objective
        return super()._generate_objective(X, y, beta, parameters, auxiliaries)


class L1L0(MixedL0):
    r"""L1L0 regularized Regressor.

    Regressor with L1L0 regularization solved with mixed integer programming
    as discussed in:

    https://arxiv.org/abs/1807.10753

    Extended to allow grouping of coefficients and group-level hierarchy as described
    in:

    https://doi.org/10.1287/opre.2015.1436

    Regularized regression objective:

    .. math::

        \min_{\beta} || X \beta - y ||^2_2 + \alpha \sum_{G} z_G + \eta ||\beta||_1

    Where G represents groups of features/coefficients and :math:`z_G` is are boolean
    valued slack variables.

    Args:
        groups (NDArray):
            1D array-like of integers specifying groups. Length should be the
            same as model, where each integer entry specifies the group
            each parameter corresponds to. If no grouping is needed pass a list
            of all distinct numbers (ie range(len(coefs)) to create singleton groups
            for each parameter.
        alpha (float):
            L0 pseudo-norm regularization hyper-parameter.
        eta (float):
            L1 regularization hyper-parameter.
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
            Whether to ignore cvxpy's PSD checks of matrix used in quadratic
            form. Default is True to avoid raising errors for poorly
            conditioned matrices. But if you want to be strict set to False.
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
    """

    def __init__(
        self,
        groups: NDArray | None = None,
        alpha: float = 1.0,
        eta: float = 1.0,
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
            eta=eta,
            alpha=alpha,
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

    def _generate_auxiliaries(
        self, X: NDArray, y: NDArray, beta: cp.Variable, parameters: SimpleNamespace
    ) -> SimpleNamespace | None:
        """Generate the boolean slack variable."""
        auxiliaries = super()._generate_auxiliaries(X, y, beta, parameters)
        X.shape[1] if self.groups is None else len(np.unique(self.groups))
        auxiliaries.z1 = cp.Variable(X.shape[1])
        return auxiliaries

    def _generate_constraints(
        self,
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
    ) -> list[cp.Constraint]:
        """Generate the constraints used to solve l1l0 regularization."""
        constraints = super()._generate_constraints(X, y, beta, parameters, auxiliaries)
        # L1 constraints (why not do an l1 norm in the objective instead?)
        constraints += [-auxiliaries.z1 <= beta, beta <= auxiliaries.z1]
        return constraints

    def _generate_objective(
        self,
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
    ) -> cp.Expression:
        """Generate the objective function used in l1l0 regression model."""
        c0 = 2 * X.shape[0]  # keeps hyperparameter scale independent
        objective = super()._generate_objective(X, y, beta, parameters, auxiliaries)
        objective += c0 * parameters.eta * cp.sum(auxiliaries.z1)
        return objective


class L2L0(TikhonovMixin, MixedL0):
    r"""L2L0 regularized Regressor.

    Based on Regressor with L2L0 regularization solved with mixed integer programming
    proposed in:

    https://arxiv.org/abs/2204.13789

    Extended to allow grouping of coefficients and group-level hierarchy as described
    in:

    https://doi.org/10.1287/opre.2015.1436

    And allows using a Tihkonov matrix in the l2 term.

    Regularized regression objective:

    .. math::

        \min_{\beta} || X \beta - y ||^2_2 + \alpha \sum_{G} z_G + \eta ||W\beta||^2_2

    Where G represents groups of features/coefficients and :math:`z_G` is are boolean
    valued slack variables. W is a Tikhonov matrix.

    Args:
        groups (NDArray):
            1D array-like of integers specifying groups. Length should be the
            same as model, where each integer entry specifies the group
            each parameter corresponds to. If no grouping is needed pass a list
            of all distinct numbers (ie range(len(coefs)) to create singleton groups
            for each parameter.
        alpha (float):
            L0 pseudo-norm regularization hyper-parameter.
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
            Wether to ignore cvxpy's PSD checks of matrix used in quadratic
            form. Default is True to avoid raising errors for poorly
            conditioned matrices. But if you want to be strict set to False.
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
            See docs in CVXEstimator for more information.

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
    """

    def __init__(
        self,
        groups: NDArray | None = None,
        alpha: float = 1.0,
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
            alpha=alpha,
            eta=eta,
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
