"""MIQP based solvers for Best Subset Selection solutions.

Allows hierarchy constraints similar to mixed L0 solvers.
"""

__author__ = "Luis Barroso-Luque"

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.wraps import psd_wrap

from sparselm.model._base import CVXEstimator


class BestSubsetSelection(CVXEstimator):
    """MIQP Best Subset Selection estimator.

    WARNING: Even with gurobi solver, this can take a very long time to
    converge for large problems and under-determined problems.
    """

    def __init__(
        self,
        sparse_bound,
        big_M=1000,
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
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

        self._bound = cp.Parameter(nonneg=True, value=sparse_bound)
        self.hierarchy = hierarchy
        self._big_M = cp.Parameter(nonneg=True, value=big_M)
        self.ignore_psd_check = ignore_psd_check
        self._z0 = None

    @property
    def sparse_bound(self):
        """Get sparse bound value."""
        return self._bound.value

    @sparse_bound.setter
    def sparse_bound(self, val):
        """Set sparse bound value."""
        if val <= 0:
            raise ValueError(f"sparse_bound must be > 0")
        self._bound.value = val

    @property
    def big_M(self):
        """Get MIQP big M value."""
        return self._big_M.value

    @big_M.setter
    def big_M(self, val):
        """Set MIQP big M value."""
        self._big_M.value = val

    def _gen_objective(self, X, y):
        """Generate the quadratic form portion of objective."""
        # psd_wrap will ignore cvxpy PSD checks, without it errors will
        # likely be raised since correlation matrices are usually very
        # poorly conditioned
        XTX = psd_wrap(X.T @ X) if self.ignore_psd_check else X.T @ X
        objective = cp.quad_form(self._beta, XTX) - 2 * y.T @ X @ self._beta
        # objective = cp.sum_squares(X @ self._beta - y)
        return objective

    def _gen_constraints(self, X, y):
        """Generate the constraints used to solve l0 regularization."""
        self._z0 = cp.Variable(X.shape[1], boolean=True)
        constraints = [
            self._big_M * self._z0 >= self._beta,
            self._big_M * self._z0 >= -self._beta,
            cp.sum(self._z0) <= self._bound,
        ]

        if self.hierarchy is not None:
            constraints += self._gen_hierarchy_constraints()
        return constraints

    def _gen_hierarchy_constraints(self):
        """Generate single feature hierarchy constraints."""
        return [
            self._z0[high_id] <= self._z0[sub_id]
            for high_id, sub_ids in enumerate(self.hierarchy)
            for sub_id in sub_ids
        ]


class RidgedBestSubsetSelection(BestSubsetSelection):
    """MIQP  Best subset selection estimator with ridge regularization."""

    def __init__(
        self,
        sparse_bound,
        alpha=1.0,
        big_M=1000,
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
        self._alpha = cp.Parameter(nonneg=True, value=alpha)

    @property
    def alpha(self):
        """Get alpha hyper-parameter value."""
        return self._alpha.value

    @alpha.setter
    def alpha(self, val):
        """Set alpha hyper-parameter value."""
        self._alpha.value = val

    def _gen_objective(self, X, y):
        """Generate the objective function used in l2l0 regression model."""
        c0 = 2 * X.shape[0]  # keeps hyperparameter scale independent
        objective = super()._gen_objective(X, y) + c0 * self._alpha * cp.sum_squares(
            self._beta
        )
        return objective


class BestGroupSelection(BestSubsetSelection):
    """MIQP Best group selection estimator."""

    def __init__(
        self,
        groups,
        sparse_bound,
        big_M=1000,
        hierarchy=None,
        ignore_psd_check=True,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
        **kwargs,
    ):
        """Initialize a Lasso estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to.
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
        self.groups = np.asarray(groups)
        self._group_masks = [self.groups == i for i in np.unique(groups)]
        self._z0 = cp.Variable(len(self._group_masks), boolean=True)

    def _gen_constraints(self, X, y):
        """Generate the constraints used to solve l0 regularization."""
        constraints = []
        for i, mask in enumerate(self._group_masks):
            constraints += [
                self._big_M * self._z0[i] >= self._beta[mask],
                self._big_M * self._z0[i] >= -self._beta[mask],
            ]

        if self.hierarchy is not None:
            constraints += self._gen_hierarchy_constraints()
        return constraints


class RidgedBestGroupSelection(RidgedBestSubsetSelection, BestGroupSelection):
    """Best group selection estimator with ridge regularization."""

    def __init__(
        self,
        groups,
        sparse_bound,
        alpha=1.0,
        big_M=1000,
        hierarchy=None,
        ignore_psd_check=True,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
    ):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to.
            sparse_bound (int):
                Upper bound on sparsity. The upper bound on total number of
                nonzero coefficients.
            alpha (float):
                Ridge regularization hyper-parameter.
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
        # need to call super for sklearn clone function
        super().__init__(
            groups=groups,
            sparse_bound=sparse_bound,
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

    def _gen_objective(self, X, y):
        RidgedBestSubsetSelection._gen_objective(self, X, y)

    def _gen_constraints(self, X, y):
        BestGroupSelection._gen_constraints(self, X, y)
