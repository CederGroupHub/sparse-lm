"""Base class for mixed-integer quadratic programming l0 pseudo norm based estimators."""


__author__ = "Luis Barroso-Luque"

from abc import ABCMeta
from numbers import Real
from types import SimpleNamespace
from typing import Optional

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.wraps import psd_wrap
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from ..._utils.validation import _check_groups
from .._base import CVXEstimator


class MIQP_L0(CVXEstimator, metaclass=ABCMeta):
    """Base class for mixed-integer quadratic programming (MIQP) estimators.

    Generalized l0 formulation that allows grouping coefficients, based on:

    https://doi.org/10.1287/opre.2015.1436
    """

    _parameter_constraints: dict = {"ignore_psd_check": ["boolean"]} | CVXEstimator._parameter_constraints
    _cvx_parameter_constraints: dict = {
    "big_M": [Interval(type=Real, left=0.0, right=None, closed="left")]
    }

    def __init__(
        self,
        groups=None,
        big_M=100,
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
                each parameter corresponds to. If no grouping is required, simply
                pass a list of all different numbers, i.e. using range.
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
            solver_options (dict):
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

        self.hierarchy = hierarchy
        self.ignore_psd_check = ignore_psd_check
        self.groups = groups
        self.big_M = big_M

    def _validate_params(self, X: ArrayLike, y: ArrayLike):
        """Validate parameters."""
        super()._validate_params(X, y)
        self.groups = _check_groups(self.groups, X.shape[1])

    def _generate_auxiliaries(
        self, X: ArrayLike, y: ArrayLike, beta: cp.Variable, parameters: SimpleNamespace
    ) -> Optional[SimpleNamespace]:
        """Generate the boolean slack variable."""
        n_groups = X.shape[1] if self.groups is None else len(np.unique(self.groups))
        return SimpleNamespace(z0=cp.Variable(n_groups, boolean=True))

    def _generate_objective(
        self,
        X: ArrayLike,
        y: ArrayLike,
        beta: cp.Variable,
        parameters: Optional[SimpleNamespace] = None,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        """Generate the quadratic form portion of objective."""
        # psd_wrap will ignore cvxpy PSD checks, without it errors will
        # likely be raised since correlation matrices are usually very
        # poorly conditioned
        XTX = psd_wrap(X.T @ X) if self.ignore_psd_check else X.T @ X
        objective = cp.quad_form(beta, XTX) - 2 * y.T @ X @ beta
        # objective = cp.sum_squares(X @ self.beta_ - y)
        return objective

    def _generate_constraints(
        self,
        X: ArrayLike,
        y: ArrayLike,
        beta: cp.Variable,
        parameters: Optional[SimpleNamespace] = None,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> list[cp.constraints]:
        """Generate the constraints used to solve l0 regularization."""
        groups = np.arange(X.shape[1]) if self.groups is None else self.groups
        group_masks = [groups == i for i in np.sort(np.unique(groups))]
        constraints = []
        for i, mask in enumerate(group_masks):
            constraints += [
                -parameters.big_M * auxiliaries.z0[i] <= beta[mask],
                beta[mask] <= parameters.big_M * auxiliaries.z0[i],
            ]

        if self.hierarchy is not None:
            constraints += self._generate_hierarchy_constraints(groups, auxiliaries.z0)

        return constraints

    def _generate_hierarchy_constraints(self, groups: ArrayLike, z0: cp.Variable):
        """Generate single feature hierarchy constraints."""
        group_ids = np.sort(np.unique(groups))
        return [
            z0[high_id] <= z0[sub_id]
            for high_id, sub_ids in zip(group_ids, self.hierarchy)
            for sub_id in sub_ids
        ]
