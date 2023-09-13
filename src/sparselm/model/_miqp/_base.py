"""Base class for mixed-integer quadratic programming l0 pseudo norm based Regressors."""

from __future__ import annotations

__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
from numbers import Real
from types import SimpleNamespace
from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.wraps import psd_wrap
from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval

from ..._utils.validation import _check_groups
from .._base import CVXRegressor


class MIQPl0(CVXRegressor, metaclass=ABCMeta):
    r"""Base class for mixed-integer quadratic programming (MIQP) Regressors.

    Generalized l0 formulation that allows grouping coefficients, based on:

    https://doi.org/10.1287/opre.2015.1436

    Args:
        groups (list or ndarray):
            array-like of integers specifying groups. Length should be the
            same as model, where each integer entry specifies the group
            each parameter corresponds to. If no grouping is required, simply
            pass a list of all different numbers, i.e. using range.
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

    _parameter_constraints: dict[str, list[Any]] = {
        "ignore_psd_check": ["boolean"],
        **CVXRegressor._parameter_constraints,
    }

    _cvx_parameter_constraints: dict[str, list[Any]] = {
        "big_M": [Interval(type=Real, left=0.0, right=None, closed="left")]
    }

    @abstractmethod  # force inspect.isabstract to return True
    def __init__(
        self,
        groups: NDArray | None = None,
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

    def _validate_params(self, X: NDArray, y: NDArray) -> None:
        """Validate parameters."""
        super()._validate_params(X, y)
        _check_groups(self.groups, X.shape[1])

    def _generate_auxiliaries(
        self, X: NDArray, y: NDArray, beta: cp.Variable, parameters: SimpleNamespace
    ) -> SimpleNamespace | None:
        """Generate the boolean slack variable."""
        n_groups = X.shape[1] if self.groups is None else len(np.unique(self.groups))
        return SimpleNamespace(z0=cp.Variable(n_groups, boolean=True))

    def _generate_objective(
        self,
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
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
        X: NDArray,
        y: NDArray,
        beta: cp.Variable,
        parameters: SimpleNamespace | None = None,
        auxiliaries: SimpleNamespace | None = None,
    ) -> list[cp.Constraint]:
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

    def _generate_hierarchy_constraints(
        self, groups: NDArray, z0: cp.Variable
    ) -> list[cp.Constraint]:
        """Generate single feature hierarchy constraints."""
        group_ids = np.sort(np.unique(groups))
        z0_index = {gid: i for i, gid in enumerate(group_ids)}
        constraints = [
            z0[z0_index[high_id]] <= z0[z0_index[sub_id]]
            for high_id, sub_ids in zip(group_ids, self.hierarchy)
            for sub_id in sub_ids
        ]
        return constraints
