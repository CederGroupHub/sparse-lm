"""A set of generalized lasso estimators.

* Lasso
* Group Lasso
* Overlap Group Lasso
* Sparse Group Lasso
* Ridged Group Lasso

Estimators follow scikit-learn interface, but use cvxpy to set up and solve
optimization problem.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

import warnings
from numbers import Real
from types import SimpleNamespace
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import sqrtm
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_scalar

from .._utils.validation import _check_group_weights, _check_groups
from ._base import CVXCanonicals, CVXEstimator


class Lasso(CVXEstimator):
    r"""
    Lasso Estimator implemented with cvxpy.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha ||\beta||_1

    """

    _cvx_parameter_constraints: dict = {
        "alpha": [Interval(type=Real, left=0.0, right=None, closed="left")]
    }

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
    ):
        """Initialize a Lasso estimator.

        Args:
            alpha (float):
                Regularization hyper-parameter.
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
        self.alpha = alpha

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        """Generate regularization term."""
        return parameters.alpha * cp.norm1(beta)

    def _generate_objective(
        self,
        X: ArrayLike,
        y: ArrayLike,
        beta: cp.Variable,
        parameters: Optional[SimpleNamespace] = None,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        # can also use cp.norm2(X @ beta - y)**2 not sure whats better
        reg = self._generate_regularization(X, beta, parameters, auxiliaries)
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ beta - y) + reg
        return objective


class GroupLasso(Lasso):
    r"""Group Lasso implementation.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha \sum_{G} w_G ||\beta_G||_2

    Where G represents groups of features/coefficients
    """

    def __init__(
        self,
        groups=None,
        alpha=1.0,
        group_weights=None,
        standardize=False,
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
                each parameter corresponds to.
            alpha (float):
                Regularization hyper-parameter.
            fit_intercept (bool):
                Whether the intercept should be estimated or not.
                If False, the data is assumed to be already centered.
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the groups given. If you need all groups
                weighted equally just pass an array of ones.
            standardize (bool): optional
                Whether to standardize the group regularization penalty using
                the feature matrix. See the following for reference:
                http://faculty.washington.edu/nrsimon/standGL.pdf
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
        self.groups = groups
        self.standardize = standardize
        self.group_weights = group_weights

        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
            **kwargs,
        )

    def _validate_params(self, X: ArrayLike, y: ArrayLike) -> None:
        """Validate group parameters."""
        super()._validate_params(X, y)
        if self.groups is None:
            warnings.warn(
                "groups has not been supplied such that the problem reduces to"
                " a simple Lasso. You should consider using that instead.",
                UserWarning,
            )
        _check_groups(self.groups, X.shape[1])
        _check_group_weights(self.group_weights, X.shape[1])

    def _set_param_values(self) -> None:
        super()._set_param_values()
        if self.group_weights is not None:
            self.canonicals_.parameters.group_weights = self.group_weights

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        parameters = super()._generate_params(X, y)
        n_groups = X.shape[1] if self.groups is None else len(np.unique(self.groups))
        group_weights = (
            np.ones(n_groups) if self.group_weights is None else self.group_weights
        )
        parameters.group_weights = group_weights
        return parameters

    @staticmethod
    def _generate_group_norms(
        X: ArrayLike,
        groups: ArrayLike,
        beta: cp.Variable,
        standardize: bool,
        parameters: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        """Generate cp.Expression of group norms."""
        group_masks = [groups == i for i in np.sort(np.unique(groups))]
        if standardize:
            group_norms = cp.hstack(
                [cp.norm2(X[:, mask] @ beta[mask]) for mask in group_masks]
            )
        else:
            group_norms = cp.hstack([cp.norm2(beta[mask]) for mask in group_masks])
        return group_norms

    def _generate_auxiliaries(
        self, X: ArrayLike, y: ArrayLike, beta: cp.Variable, parameters: SimpleNamespace
    ) -> Optional[SimpleNamespace]:
        """Generate auxiliary cp.Expression for group norms"""
        groups = np.arange(X.shape[1]) if self.groups is None else self.groups
        group_norms = self._generate_group_norms(
            X, groups, beta, self.standardize, parameters
        )
        return SimpleNamespace(group_norms=group_norms)

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ):
        return parameters.alpha * (parameters.group_weights @ auxiliaries.group_norms)


# TODO this implementation is not efficient, reimplement.
class OverlapGroupLasso(GroupLasso):
    r"""Overlap Group Lasso implementation.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha \sum_{G} w_G ||\beta_G||_2

    Where G represents groups of features/coefficients, and overlapping groups
    are acceptable. Meaning a coefficients can be in more than one group.
    """

    def __init__(
        self,
        group_list=None,
        alpha=1.0,
        group_weights=None,
        standardize=False,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
        **kwargs,
    ):
        """Initialize estimator.

        Args:
            group_list (list of lists):
                list of lists of integers specifying groups. The length of the
                list holding lists should be the same as model. Each inner list
                has integers specifying the groups the coefficient for that
                index belongs to. i.e. [[1,2],[2,3],[1,2,3]] means the first
                coefficient belongs to group 1 and 2, the second to 2, and 3
                and the third to 1, 2 and 3. In other words the 3 groups would
                be: (0, 2), (0, 1, 2), (1, 2)
            alpha (float):
                Regularization hyper-parameter.
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the number of different groups given.
                If you need all groups weighted equally just pass an array of
                ones.
            standardize (bool): optional
                Whether to standardize the group regularization penalty using
                the feature matrix. See the following for reference:
                http://faculty.washington.edu/nrsimon/standGL.pdf
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
        self.group_list = group_list

        super().__init__(
            groups=None,
            alpha=alpha,
            group_weights=group_weights,
            standardize=standardize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
            **kwargs,
        )

    def _validate_params(self, X: ArrayLike, y: ArrayLike) -> None:
        """Validate group parameters."""
        # skip group lasso validation
        super(GroupLasso, self)._validate_params(X, y)

        if self.group_list is not None:
            if len(self.group_list) != X.shape[1]:
                raise ValueError(
                    "The length of the group list must be the same as the number of features."
                )

            n_groups = len(np.unique([gid for grp in self.group_list for gid in grp]))
        else:
            warnings.warn(
                "No group list has been supplied such that the problem reduces to"
                " a simple Lasso. You should consider using that instead.",
                UserWarning,
            )
            n_groups = X.shape[1]

        _check_group_weights(self.group_weights, n_groups)

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        parameters = super()._generate_params(X, y)

        if self.group_list is None:
            n_groups = X.shape[1]
        else:
            n_groups = len(np.unique([gid for grp in self.group_list for gid in grp]))

        group_weights = (
            np.ones(n_groups) if self.group_weights is None else self.group_weights
        )
        parameters.group_weights = group_weights
        return parameters

    def _initialize_problem(self, X: ArrayLike, y: ArrayLike):
        """Initialize cvxpy problem from the generated objective function.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector
        """

        # need to generate the auxiliaries here since the problem data is "augmented"
        # based on them
        if self.group_list is None:
            group_list = [[i] for i in range(X.shape[1])]
        else:
            group_list = self.group_list

        group_ids = np.sort(np.unique([gid for grp in group_list for gid in grp]))
        beta_indices = [
            [i for i, grp in enumerate(group_list) if grp_id in grp]
            for grp_id in group_ids
        ]
        extended_groups = np.concatenate(
            [
                len(g)
                * [
                    i,
                ]
                for i, g in enumerate(beta_indices)
            ]
        )
        beta_indices = np.concatenate(beta_indices)

        X_ext = X[:, beta_indices]
        beta = cp.Variable(X_ext.shape[1])
        parameters = self._generate_params(X, y)
        group_norms = self._generate_group_norms(
            X_ext, extended_groups, beta, self.standardize
        )
        auxiliaries = SimpleNamespace(
            group_norms=group_norms, extended_coef_indices=beta_indices
        )
        objective = self._generate_objective(X, y, beta, parameters, auxiliaries)
        constraints = self._generate_constraints(X, y, parameters, auxiliaries)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        self.canonicals_ = CVXCanonicals(
            problem=problem,
            objective=objective,
            beta=beta,
            parameters=parameters,
            auxiliaries=auxiliaries,
            constraints=constraints,
        )

    def _solve(self, X, y, solver_options, *args, **kwargs):
        """Solve the cvxpy problem."""
        self.canonicals_.problem.solve(
            solver=self.solver, warm_start=self.warm_start, **solver_options
        )
        beta = np.array(
            [
                sum(
                    self.canonicals_.beta.value[
                        self.canonicals_.auxiliaries.extended_coef_indices == i
                    ]
                )
                for i in range(X.shape[1])
            ]
        )
        return beta


class SparseGroupLasso(GroupLasso):
    r"""Sparse Group Lasso.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2
            + \alpha r ||\beta||_1
            + \alpha (1 - r) * \sum_{G}||\beta_G||_2

    Where G represents groups of features / coefficients. And r is the L1 ratio.
    """

    def __init__(
        self,
        groups=None,
        l1_ratio=0.5,
        alpha=1.0,
        group_weights=None,
        standardize=False,
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
                each parameter corresponds to.
            l1_ratio (float):
                Mixing parameter between l1 and group lasso regularization.
            alpha (float):
                Regularization hyper-parameter.
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the groups given. If you need all groups
                weighted equally just pass an array of ones.
            standardize (bool): optional
                Whether to standardize the group regularization penalty using
                the feature matrix. See the following for reference:
                http://faculty.washington.edu/nrsimon/standGL.pdf
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
            groups=groups,
            alpha=alpha,
            group_weights=group_weights,
            standardize=standardize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
            **kwargs,
        )
        self.l1_ratio = l1_ratio

    def _validate_params(self, X, y):
        """Validate parameters."""
        super()._validate_params(X, y)
        check_scalar(self.l1_ratio, "l1_ratio", float, min_val=0, max_val=1)

        if self.l1_ratio == 0.0:
            warnings.warn(
                "It is more efficient to use GroupLasso directly than SparseGroupLasso with l1_ratio=0",
                UserWarning,
            )
        if self.l1_ratio == 1.0:
            warnings.warn(
                "It is more efficient to use Lasso directly than SparseGroupLasso with l1_ratio=1",
                UserWarning,
            )

    def _set_param_values(self) -> None:
        super()._set_param_values()
        self.canonicals_.parameters.lambda1.value = self.l1_ratio * self.alpha
        self.canonicals_.parameters.lambda2.value = (1 - self.l1_ratio) * self.alpha

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        """Generate parameters."""
        parameters = super()._generate_params(X, y)
        # del parameters.alpha  # hacky but we don't need this
        parameters.l1_ratio = self.l1_ratio  # save simply for infomation
        parameters.lambda1 = cp.Parameter(nonneg=True, value=self.l1_ratio * self.alpha)
        parameters.lambda2 = cp.Parameter(
            nonneg=True, value=(1 - self.l1_ratio) * self.alpha
        )
        return parameters

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ):
        group_regularization = parameters.lambda2 * (
            parameters.group_weights @ auxiliaries.group_norms
        )
        l1_regularization = parameters.lambda1 * cp.norm1(beta)
        return group_regularization + l1_regularization


class RidgedGroupLasso(GroupLasso):
    r"""Ridged Group Lasso implementation.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha \sum_{G} w_G ||\beta_G||_2
                               + \sum_{G} \delta_l ||\beta_G||^2_2

    Where G represents groups of features/coefficients

    For details on proper standardization refer to:
    http://faculty.washington.edu/nrsimon/standGL.pdf
    """

    _cvx_parameter_constraints: dict = {
        "alpha": [Interval(type=Real, left=0.0, right=None, closed="left")],
        "delta": ["array-like", Interval(type=Real, left=0.0, right=None, closed="left")]
    }

    def __init__(
        self,
        groups=None,
        alpha=1.0,
        delta=(1.0, ),  # TODO type this as ArrayLike
        group_weights=None,
        standardize=False,
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
                each parameter corresponds to.
            alpha (float):
                Regularization hyper-parameter.
            delta (ndarray): optional
                Positive 1D array. Regularization vector for ridge penalty. The array
                must be of the same lenght as the number of groups, or length 1 if all
                groups are ment to have the same ridge hyperparamter.
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the groups given. If you need all groups
                weighted equally just pass an array of ones.
            standardize (bool): optional
                Whether to standardize the group regularization penalty using
                the feature matrix. See the following for reference:
                http://faculty.washington.edu/nrsimon/standGL.pdf
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
            groups=groups,
            alpha=alpha,
            group_weights=group_weights,
            standardize=standardize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
            **kwargs,
        )

        self.delta = delta

    def _validate_params(self, X: ArrayLike, y: ArrayLike) -> None:
        """Validate group parameters and delta."""
        super()._validate_params(X, y)
        n_groups = (
            len(np.unique(self.groups)) if self.groups is not None else X.shape[1]
        )
        if len(self.delta) != n_groups and len(self.delta) != 1:
                raise ValueError(
                    f"delta must be an array of length 1 or equal to the number of groups {n_groups}."
                )

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        """Generate parameters."""
        parameters = super()._generate_params(X, y)
        # force cvxpy delta to be an array of n_groups!
        n_groups = (
            len(np.unique(self.groups)) if self.groups is not None else X.shape[1]
        )
        if len(self.delta) != n_groups:
            delta = self.delta * np.ones(n_groups)
            parameters.delta = cp.Parameter(shape=(n_groups,), nonneg=True, value=delta)
        return parameters

    @staticmethod
    def _generate_group_norms(
        X: ArrayLike,
        groups: ArrayLike,
        beta: cp.Variable,
        standardize: bool,
        parameters: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        group_masks = [groups == i for i in np.sort(np.unique(groups))]
        if standardize:
            group_norms = cp.hstack(
                [
                    cp.norm2(
                        sqrtm(
                            X[:, mask].T @ X[:, mask]
                            + parameters.delta.value[i] ** 0.5 * np.eye(sum(mask))
                        )
                        @ beta[mask]
                    )
                    for i, mask in enumerate(group_masks)
                ]
            )
        else:
            group_norms = cp.hstack([cp.norm2(beta[mask]) for mask in group_masks])
        # self._group_norms = grp_norms.T
        return group_norms

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ):
        # repetitive code...
        groups = np.arange(X.shape[1]) if self.groups is None else self.groups
        group_masks = [groups == i for i in np.sort(np.unique(groups))]
        ridge = cp.hstack([cp.sum_squares(beta[mask]) for mask in group_masks])
        group_regularization = (
            parameters.alpha * parameters.group_weights @ auxiliaries.group_norms
        )
        ridge_regularization = 0.5 * parameters.delta @ ridge
        return group_regularization + ridge_regularization
