"""MIQP based solvers for Best Subset Selection solutions.

Allows hierarchy constraints similar to mixed L0 solvers.
"""

__author__ = "Luis Barroso-Luque"


import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
from .base import CVXEstimator


class BestSubsetSelection(CVXEstimator):
    """MIQP Best Subset Selection estimator

    WARNING: Even with gurobi solver, this can take a very long time to
    converge for large problems and underdetermined problems.
    """

    def __init__(self, sparse_bound, big_M=1000, hierarchy=None,
                 ignore_psd_check=True, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """

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
                Wether to ignore cvxpy's PSD checks  of matrix used in quadratic
                form. Default is True to avoid raising errors for poorly
                conditioned matrices. But if you want to be strict set to False.
            fit_intercept (bool):
                Whether the intercept should be estimated or not.
                If False, the data is assumed to be already centered.
            normalize (bool):
                This parameter is ignored when fit_intercept is set to False.
                If True, the regressors X will be normalized before regression
                by subtracting the mean and dividing by the l2-norm.
                If you wish to standardize, please use StandardScaler before
                calling fit on an estimator with normalize=False
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
            **kwargs:
                Kewyard arguments passed to cvxpy solve.
                See docs linked above for more information.
        """
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, warm_start=warm_start, solver=solver,
                         **kwargs)

        self._bound = cp.Parameter(nonneg=True, value=sparse_bound)
        self.hierarchy = hierarchy
        self._big_M = cp.Parameter(nonneg=True, value=big_M)
        self.ignore_psd_check = ignore_psd_check
        self._z0 = None

    @property
    def sparse_bound(self):
        return self._bound.value

    @sparse_bound.setter
    def sparse_bound(self, val):
        if val <= 0:
            raise ValueError(f"sparse_bound must be > 0")
        self._bound.value = val

    @property
    def big_M(self):
        return self._big_M.value

    @big_M.setter
    def big_M(self, val):
        self._big_M.value = val

    def _gen_objective(self, X, y):
        """Generate the quadratic form portion of objective"""
        # psd_wrap will ignore cvxpy PSD checks, without it errors will
        # likely be raised since correlation matrices are usually very
        # poorly conditioned
        XTX = psd_wrap(X.T @ X) if self.ignore_psd_check else X.T @ X
        objective = cp.quad_form(self._beta, XTX) - 2 * y.T @ X @ self._beta
        # objective = cp.sum_squares(X @ self._beta - y)
        return objective

    def _gen_constraints(self, X, y):
        """Generate the constraints used to solve l0 regularization"""
        self._z0 = cp.Variable(X.shape[1], boolean=True)
        constraints = [self._big_M * self._z0 >= self._beta,
                       self._big_M * self._z0 >= -self._beta,
                       cp.sum(self._z0) <= self._bound]

        if self.hierarchy is not None:
            constraints += self._gen_hierarchy_constraints()
        return constraints

    def _gen_hierarchy_constraints(self):
        """Generate single feature hierarchy constraints"""
        return [self._z0[high_id] <= self._z0[sub_id]
                for high_id, sub_ids in enumerate(self.hierarchy)
                for sub_id in sub_ids]
