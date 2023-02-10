"""MIQP based solvers for Best Subset Selection solutions.

Allows hierarchy constraints similar to mixed L0 solvers.
"""

__author__ = "Luis Barroso-Luque"

import cvxpy as cp

from sparselm.model._base import TikhonovMixin

from ._base import MIQP_L0


class BestSubsetSelection(MIQP_L0):
    """MIQP Best Subset Selection estimator.

    Generalized best subset that allows grouping subsets.

    WARNING: Even with gurobi solver, this can take a very long time to
    converge for large problems and under-determined problems.
    """

    def __init__(
        self,
        groups=None,
        sparse_bound=100,
        big_M=100.0,
        hierarchy=None,
        ignore_psd_check=True,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
        **kwargs,
    ):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to. If no grouping is required, simply
                pass a list of all different numbers, i.e. using range.
            sparse_bound (int):
                Upper bound on sparsity. The upper bound on total number of
                nonzero coefficients.
            big_M (float):
                Upper bound on the norm of coefficients associated with each
                cluster (groups of coefficients) ||Beta_c||_2
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
                cvxpy backend solver to use. Supported solvers are:
                ECOS, ECOS_BB, CVXOPT, SCS, GUROBI, Elemental.
                GLPK and GLPK_MI (via CVXOPT GLPK interface)
            solver_options:
                dictionary of keyword arguments passed to cvxpy solve.
                See docs in CVXEstimator for more information.
        """
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
        if sparse_bound <= 0:
            raise ValueError("sparse_bound must be > 0")

        self._bound = cp.Parameter(nonneg=True, value=sparse_bound)

    @property
    def sparse_bound(self):
        """Get sparse bound value."""
        return self._bound.value

    @sparse_bound.setter
    def sparse_bound(self, val):
        """Set sparse bound value."""
        if val <= 0:
            raise ValueError("sparse_bound must be > 0")
        self._bound.value = val

    def _gen_constraints(self, X, y):
        """Generate the constraints used to solve l0 regularization."""
        constraints = [cp.sum(self._z0) <= self._bound]
        constraints += super()._gen_constraints(X, y)
        return constraints


class RidgedBestSubsetSelection(TikhonovMixin, BestSubsetSelection):
    """MIQP  Best subset selection estimator with ridge regularization."""

    def __init__(
        self,
        groups=None,
        sparse_bound=100,
        eta=1.0,
        big_M=100.0,
        hierarchy=None,
        tikhonov_w=None,
        ignore_psd_check=True,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
        **kwargs,
    ):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to. If no grouping is required, simply
                pass a list of all different numbers, i.e. using range.
            sparse_bound (int):
                Upper bound on sparsity. The upper bound on total number of
                nonzero coefficients.
            eta (float):
                L2 regularization hyper-parameter.
            big_M (float):
                Upper bound on the norm of coefficients associated with each
                cluster (groups of coefficients) ||Beta_c||_2
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
                cvxpy backend solver to use. Supported solvers are:
                ECOS, ECOS_BB, CVXOPT, SCS, GUROBI, Elemental.
                GLPK and GLPK_MI (via CVXOPT GLPK interface)
            solver_options:
                dictionary of keyword arguments passed to cvxpy solve.
                See docs in CVXEstimator for more information.
        """
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
            **kwargs,
        )
        self.tikhonov_w = tikhonov_w
        self._eta = cp.Parameter(nonneg=True, value=eta)

    @property
    def eta(self):
        """Get alpha hyper-parameter value."""
        return self._eta.value

    @eta.setter
    def eta(self, val):
        """Set alpha hyper-parameter value."""
        self._eta.value = val
