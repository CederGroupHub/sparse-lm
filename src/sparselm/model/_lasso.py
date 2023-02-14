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
from types import SimpleNamespace
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import sqrtm
from sklearn.utils.validation import check_scalar

from .._utils.validation import _check_group_weights, _check_groups
from ._base import CVXEstimator


class Lasso(CVXEstimator):
    r"""
    Lasso Estimator implemented with cvxpy.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha ||\beta||_1

    """

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
        self.alpha = alpha
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

    def _validate_params(self, X: ArrayLike, y: ArrayLike) -> None:
        """Validate parameters."""
        super()._validate_params(X, y)
        check_scalar(self.alpha, "alpha", float, min_val=0.0)

    def _set_param_values(self) -> None:
        """Set parameter values."""
        self.canonicals_.parameters.alpha.value = self.alpha

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        """Generate cvxpy parameters."""
        return SimpleNamespace(alpha=cp.Parameter(nonneg=True, value=self.alpha))

    def _generate_regularization(
        self, X: ArrayLike, beta: cp.Variable, parameters: SimpleNamespace,
            auxiliaries: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        """Generate regularization term."""
        return parameters.alpha * cp.norm1(beta)

    def _generate_objective(
        self,
        X: ArrayLike,
        y: ArrayLike,
        beta: cp.Variable,
        auxiliaries: Optional[SimpleNamespace] = None,
        parameters: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        # can also use cp.norm2(X @ beta - y)**2 not sure whats better
        reg = self._generate_regularization(X, beta, parameters)
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
        self.groups = _check_groups(self.groups, X.shape[1])
        self.group_weights = _check_group_weights(self.group_weights, self.groups)

    def _generate_auxiliaries(
            self, X: ArrayLike, y: ArrayLike, beta: cp.Variable, parameters: SimpleNamespace
    ) -> Optional[SimpleNamespace]:
        """Generate auxiliary cp.Expression for group norms"""
        group_masks = [self.groups == i for i in np.sort(np.unique(self.groups))]
        if self.standardize:
            group_norms = cp.hstack(
                [cp.norm2(X[:, mask] @ beta[mask]) for mask in group_masks]
            )
        else:
            group_norms = cp.hstack([cp.norm2(beta[mask]) for mask in group_masks])
        return SimpleNamespace(group_norms=group_norms)

    def _generate_regularization(
            self, X: ArrayLike, beta: cp.Variable, parameters: SimpleNamespace,
            auxiliaries: Optional[SimpleNamespace] = None,
    ):
        return parameters.alpha * (self.group_weights @ auxiliaries.group_norms)


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
        group_list,
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

    def _validate_params(self, X, y):
        """Validate group parameters."""
        Lasso._validate_params(self, X, y)
        check_scalar(self.alpha, "alpha", float, min_val=0.0)
        if len(self.group_list) != X.shape[1]:
            raise ValueError(
                "The length of the group list must be the same as the number of features."
            )

        group_ids = np.sort(np.unique([gid for grp in self.group_list for gid in grp]))
        beta_indices = [
            [i for i, grp in enumerate(self.group_list) if grp_id in grp]
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
        self.groups = _check_groups(extended_groups, len(extended_groups))
        self.group_weights = _check_group_weights(self.group_weights, self.groups)
        self.ext_coef_indices_ = np.concatenate(beta_indices)

    def _initialize_problem(self, X: ArrayLike, y: ArrayLike):
        """Initialize cvxpy problem from the generated objective function.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector
        """
        X_ext = X[:, self.ext_coef_indices_]
        self.beta_ = cp.Variable(X_ext.shape[1])
        self.objective_ = self._generate_objective(X_ext, y)
        self.constraints_ = self._generate_constraints(X_ext, y)
        self.problem_ = cp.Problem(cp.Minimize(self.objective_), self.constraints_)

    def _solve(self, X, y, solver_options, *args, **kwargs):
        """Solve the cvxpy problem."""
        self.problem_.solve(
            solver=self.solver, warm_start=self.warm_start, **solver_options
        )
        beta = np.array(
            [
                sum(self.beta_.value[self.ext_coef_indices_ == i])
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
        self.lambda1_ = cp.Parameter(nonneg=True, value=l1_ratio * alpha)
        self.lambda2_ = cp.Parameter(nonneg=True, value=(1 - l1_ratio) * alpha)

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

    def _generate_params(self):
        """Generate parameters."""
        if not hasattr(self, "lambda1_"):
            self.lambda1_ = cp.Parameter(nonneg=True, value=self.l1_ratio * self.alpha)
        else:
            self.lambda1_.value = self.l1_ratio * self.alpha

        if not hasattr(self, "lambda2_"):
            self.lambda2_ = cp.Parameter(
                nonneg=True, value=(1 - self.l1_ratio) * self.alpha
            )
        else:
            self.lambda2_.value = (1 - self.l1_ratio) * self.alpha

    def _generate_regularization(self, X):
        grp_norms = super()._generate_auxiliaries(X)
        l1_reg = cp.norm1(self.beta_)
        reg = self.lambda1_ * l1_reg + self.lambda2_ * (self.group_weights @ grp_norms)
        return reg


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

    def __init__(
        self,
        groups=None,
        alpha=1.0,
        delta=1.0,
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
                Positive 1D array. Regularization vector for ridge penalty.
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

    def _generate_params(self):
        """Generate parameters."""
        super()._generate_params()

        n_groups = len(np.unique(self.groups))
        if isinstance(self.delta, float):
            delta = self.delta * np.ones(n_groups)
        else:
            delta = self.delta

        if not hasattr(self, "delta_"):
            self.delta_ = cp.Parameter(shape=(n_groups,), nonneg=True, value=delta)
        else:
            self.delta_.value = delta

    def _generate_auxiliaries(self, X):
        # TODO remove this, see above TODO
        self.group_masks_ = [self.groups == i for i in np.sort(np.unique(self.groups))]

        if self.standardize:
            grp_norms = cp.hstack(
                [
                    cp.norm2(
                        sqrtm(
                            X[:, mask].T @ X[:, mask]
                            + self.delta_.value[i] ** 0.5 * np.eye(sum(mask))
                        )
                        @ self.beta_[mask]
                    )
                    for i, mask in enumerate(self.group_masks_)
                ]
            )
        else:
            grp_norms = cp.hstack(
                [cp.norm2(self.beta_[mask]) for mask in self.group_masks_]
            )

        self._group_norms = grp_norms.T
        return grp_norms

    def _generate_regularization(self, X):
        self._generate_params()
        grp_norms = self._generate_auxiliaries(X)
        ridge = cp.hstack(
            [cp.sum_squares(self.beta_[mask]) for mask in self.group_masks_]
        )
        reg = self.alpha_ * self.group_weights @ grp_norms + 0.5 * self.delta_ @ ridge

        return reg
