"""A set of generalized adaptive lasso estimators.

* Adaptive Lasso
* Adaptive Group Lasso
* Adaptive Overlap Group Lasso
* Adaptive Sparse Group Lasso
* Adaptive Ridged Group Lasso

Estimators follow scikit-learn interface, but use cvxpy to set up and solve
optimization problem.

NOTE: In certain cases these can yield infeasible problems. This can cause
processes to die and as a result make a calculation hang indefinitely when
using them in a multiprocess model selection tool such as sklearn
GridSearchCV with n_jobs > 1.

In that case either tweak settings/solvers around so that that does not happen
or run with n_jobs=1 (but that may take a while to solve)
"""

__author__ = "Luis Barroso-Luque"

import warnings
from types import SimpleNamespace
from typing import Optional

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_scalar

from ._lasso import (
    GroupLasso,
    Lasso,
    OverlapGroupLasso,
    RidgedGroupLasso,
    SparseGroupLasso,
)


class AdaptiveLasso(Lasso):
    r"""Adaptive Lasso implementation.

    Also known as iteratively re-weighted Lasso.
    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha ||w^T \beta||_1

    Where w represents a vector of weights that is iteratively updated.
    """

    def __init__(
        self,
        alpha=1.0,
        max_iter=3,
        eps=1e-6,
        tol=1e-10,
        update_function=None,
        fit_intercept=False,
        copy_X=True,
        warm_start=True,
        solver=None,
        solver_options=None,
        **kwargs,
    ):
        """Initialize estimator.

        Args:
            alpha (float):
                Regularization hyper-parameter.
            max_iter (int):
                Maximum number of re-weighting iteration steps.
            eps (float):
                Value to add to denominator of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
            update_function (Callable): optional
                A function with signature f(beta, eps) used to update the
                weights at each iteration. Default is 1/(|beta| + eps)
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
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
            **kwargs,
        )
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        self.update_function = update_function

    def _validate_params(self, X: ArrayLike, y: ArrayLike) -> None:
        super()._validate_params(X, y)
        check_scalar(self.max_iter, "max_iter", int, min_val=1)
        check_scalar(self.eps, "eps", float)
        check_scalar(self.tol, "tol", float)

        if self.max_iter == 1:
            warnings.warn(
                "max_iter is set to 1. It should ideally be set > 1, otherwise consider "
                "using a non-adaptive estimator",
                UserWarning,
            )

        if self.update_function is not None:
            if not callable(self.update_function):
                raise ValueError("update_function must be callable.")

    def _set_param_values(self) -> None:
        """Set parameter values."""
        length = len(self.canonicals_.parameters.adaptive_weights.value)
        self.canonicals_.parameters.adaptive_weights.value = self.alpha * np.ones(
            length
        )

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        """Generate parameters for the problem."""
        parameters = super()._generate_params(X, y)
        parameters.adaptive_weights = cp.Parameter(
            shape=X.shape[1], nonneg=True, value=self.alpha * np.ones(X.shape[1])
        )
        return parameters

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        """Generate regularization term."""
        return cp.norm1(cp.multiply(parameters.adaptive_weights, beta))

    def _get_update_function(self):
        if self.update_function is None:
            return lambda beta, eps: self.alpha / (abs(beta) + eps)
        return self.update_function

    @staticmethod
    def _get_weights_value(parameters: SimpleNamespace):
        """Simply return a copy of the value of adaptive weights."""
        return parameters.adaptive_weights.value.copy()

    def _check_convergence(
        self, parameters: SimpleNamespace, previous_weights: ArrayLike
    ):
        """Check if weights have converged to set tolerance."""
        current_weights = parameters.adaptive_weights.value
        return np.linalg.norm(current_weights - previous_weights) <= self.tol

    def _iterative_update(
        self,
        beta: ArrayLike,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> None:
        """Update the adaptive weights."""
        update = self._get_update_function()
        parameters.adaptive_weights.value = self.alpha * update(beta, self.eps)

    def _solve(self, X: ArrayLike, y: ArrayLike, solver_options: dict, *args, **kwargs):
        """Solve Lasso problem iteratively adaptive weights."""
        previous_weights = self._get_weights_value(self.canonicals_.parameters)
        for i in range(self.max_iter):
            if (
                self.canonicals_.beta.value is None
                and self.canonicals_.problem.value == -np.inf
            ):
                raise RuntimeError(f"{self.canonicals_.problem} is infeasible.")
            self.canonicals_.problem.solve(
                solver=self.solver, warm_start=self.warm_start, **solver_options
            )
            self.n_iter_ = i + 1  # save number of iterations for sklearn
            self._iterative_update(
                self.canonicals_.beta.value,
                self.canonicals_.parameters,
                self.canonicals_.auxiliaries,
            )
            # check convergence
            if self._check_convergence(self.canonicals_.parameters, previous_weights):
                break
            previous_weights = self._get_weights_value(self.canonicals_.parameters)
        return self.canonicals_.beta.value


class AdaptiveGroupLasso(AdaptiveLasso, GroupLasso):
    r"""Adaptive Group Lasso, iteratively re-weighted group lasso.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha * \sum_{G} w_G ||\beta_G||_2

    Where w represents a vector of weights that is iteratively updated.
    """

    def __init__(
        self,
        groups=None,
        alpha=1.0,
        group_weights=None,
        max_iter=3,
        eps=1e-6,
        tol=1e-10,
        update_function=None,
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
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the groups given. If you need all groups
                weighted equally just pass an array of ones.
            max_iter (int):
                Maximum number of re-weighting iteration steps.
            eps (float):
                Value to add to denominator of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
            update_function (Callable): optional
                A function with signature f(group_norms, eps) used to update the
                weights at each iteration. Where group_norms are the norms of
                the coefficients Beta for each group.
                Default is 1/(group_norms + eps)
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
        # call with keywords to avoid MRO issues
        super().__init__(
            groups=groups,
            alpha=alpha,
            group_weights=group_weights,
            max_iter=max_iter,
            eps=eps,
            tol=tol,
            update_function=update_function,
            standardize=standardize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
            **kwargs,
        )

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        # skip AdaptiveLasso in super
        parameters = super(AdaptiveLasso, self)._generate_params(X, y)
        n_groups = X.shape[1] if self.groups is None else len(np.unique(self.groups))
        parameters.adaptive_weights = cp.Parameter(
            shape=n_groups,
            nonneg=True,
            value=self.alpha * np.ones(n_groups),
        )
        return parameters

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ):
        return parameters.adaptive_weights @ auxiliaries.group_norms

    def _iterative_update(
        self,
        beta: ArrayLike,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> None:
        update = self._get_update_function()
        parameters.adaptive_weights.value = (
            self.alpha * parameters.group_weights
        ) * update(auxiliaries.group_norms.value, self.eps)


class AdaptiveOverlapGroupLasso(OverlapGroupLasso, AdaptiveGroupLasso):
    r"""Adaptive Overlap Group Lasso implementation.

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
        max_iter=3,
        eps=1e-6,
        tol=1e-10,
        update_function=None,
        standardize=False,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
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
            max_iter (int):
                Maximum number of re-weighting iteration steps.
            eps (float):
                Value to add to denominator of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
            update_function (Callable): optional
                A function with signature f(group_norms, eps) used to update the
                weights at each iteration. Where group_norms are the norms of
                the coefficients Beta for each group.
                Default is 1/(group_norms + eps)
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
        # call with keywords to avoid MRO issues
        super().__init__(
            group_list=group_list,
            alpha=alpha,
            group_weights=group_weights,
            max_iter=max_iter,
            eps=eps,
            tol=tol,
            update_function=update_function,
            standardize=standardize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

    def _generate_objective(
        self,
        X: ArrayLike,
        y: ArrayLike,
        beta: cp.Variable,
        parameters: Optional[SimpleNamespace] = None,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        return AdaptiveGroupLasso._generate_objective(
            self, X, y, beta, parameters, auxiliaries
        )

    def _solve(self, X: ArrayLike, y: ArrayLike, solver_options: dict, *args, **kwargs):
        extended_indices = self.canonicals_.auxiliaries.extended_coef_indices
        beta = AdaptiveGroupLasso._solve(
            self, X[:, extended_indices], y, solver_options, *args, **kwargs
        )
        beta = np.array([sum(beta[extended_indices == i]) for i in range(X.shape[1])])
        return beta


class AdaptiveSparseGroupLasso(AdaptiveLasso, SparseGroupLasso):
    r"""Adaptive Sparse Group Lasso, iteratively re-weighted sparse group lasso.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2
            + \alpha r ||w^T \beta||_1
            + \alpha (1 - r) \sum_{G} v_G ||\beta_G||_2

    Where w, v represent vectors of weights that are iteratively updated.
    And r is the L1 ratio.
    """

    def __init__(
        self,
        groups=None,
        l1_ratio=0.5,
        alpha=1.0,
        group_weights=None,
        max_iter=3,
        eps=1e-6,
        tol=1e-10,
        update_function=None,
        standardize=False,
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
            max_iter (int):
                Maximum number of re-weighting iteration steps.
            eps (float):
                Value to add to denominator of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
            update_function (Callable): optional
                A function with signature f(group_norms, eps) used to update the
                weights at each iteration. Where group_norms are the norms of
                the coefficients Beta for each group.
                Default is 1/(group_norms + eps)
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
        # call with keywords to avoid MRO issues
        super().__init__(
            groups=groups,
            l1_ratio=l1_ratio,
            alpha=alpha,
            group_weights=group_weights,
            max_iter=max_iter,
            eps=eps,
            tol=tol,
            update_function=update_function,
            standardize=standardize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

    def _set_param_values(self) -> None:
        super()._set_param_values()
        group_weights = self.canonicals_.parameters.adaptive_group_weights.value
        group_weights = self.canonicals_.parameters.lambda1.value * np.ones_like(
            group_weights
        )
        self.canonicals_.parameters.adaptive_group_weights.value = group_weights
        coef_weights = self.canonicals_.parameters.adaptive_coef_weights.value
        coef_weights = self.canonicals_.parameters.lambda1.value * np.ones_like(
            coef_weights
        )
        self.canonicals_.parameters.adaptive_group_weights.value = coef_weights

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        # skip AdaptiveLasso in super
        parameters = super(AdaptiveLasso, self)._generate_params(X, y)
        n_groups = X.shape[1] if self.groups is None else len(np.unique(self.groups))
        parameters.adaptive_coef_weights = cp.Parameter(
            shape=X.shape[1],
            nonneg=True,
            value=parameters.lambda1.value * np.ones(X.shape[1]),
        )
        parameters.adaptive_group_weights = cp.Parameter(
            shape=n_groups,
            nonneg=True,
            value=parameters.lambda2.value * np.ones(n_groups),
        )
        return parameters

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ):
        group_regularization = (
            parameters.adaptive_group_weights @ auxiliaries.group_norms
        )
        l1_regularization = cp.norm1(
            cp.multiply(parameters.adaptive_coef_weights, beta)
        )
        return group_regularization + l1_regularization

    @staticmethod
    def _get_weights_value(parameters: SimpleNamespace):
        """Simply return a copy of the value of adaptive weights."""
        # does concatenate copy?
        concat_weights = np.concatenate(
            (
                parameters.adaptive_group_weights.value.copy(),
                parameters.adaptive_coef_weights.value.copy(),
            )
        )
        return concat_weights

    def _check_convergence(
        self, parameters: SimpleNamespace, previous_weights: ArrayLike
    ):
        """Check if weights have converged to set tolerance."""
        # This will technically check the norm of the concatenation instead of the sum
        # of the norm of each weight vector, so it's a bit of tighter tolerance.
        current_weights = np.concatenate(
            (
                parameters.adaptive_group_weights.value,
                parameters.adaptive_coef_weights.value,
            )
        )
        return np.linalg.norm(current_weights - previous_weights) <= self.tol

    def _iterative_update(
        self,
        beta: ArrayLike,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ) -> None:
        update = self._get_update_function()
        parameters.adaptive_coef_weights.value = (
            self.canonicals_.parameters.lambda1.value * update(beta, self.eps)
        )
        parameters.adaptive_group_weights.value = (
            self.canonicals_.parameters.lambda2.value * parameters.group_weights
        ) * update(auxiliaries.group_norms.value, self.eps)


class AdaptiveRidgedGroupLasso(AdaptiveGroupLasso, RidgedGroupLasso):
    r"""Adaptive Ridged Group Lasso implementation.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha \sum_{G} w_G ||\beta_G||_2
                               + \sum_{G} w_l ||\beta_G||^2_2

    Where G represents groups of features/coefficients, and w_l represents a
    vector of weights that are updated iteratively.

    For details on proper standardization refer to:
    http://faculty.washington.edu/nrsimon/standGL.pdf

    * Adaptive iterative weights are only done on the group norm and not the ridge
    portion.
    """

    def __init__(
        self,
        groups=None,
        alpha=1.0,
        delta=(1.0, ),
        group_weights=None,
        max_iter=3,
        eps=1e-6,
        tol=1e-10,
        update_function=None,
        standardize=False,
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
            fit_intercept (bool):
                Whether the intercept should be estimated or not.
                If False, the data is assumed to be already centered.
            max_iter (int):
                Maximum number of re-weighting iteration steps.
            eps (float):
                Value to add to denominator of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
            update_function (Callable): optional
                A function with signature f(group_norms, eps) used to update the
                weights at each iteration. Where group_norms are the norms of
                the coefficients Beta for each group.
                Default is 1/(group_norms + eps)
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
            delta=delta,
            max_iter=max_iter,
            eps=eps,
            tol=tol,
            update_function=update_function,
            group_weights=group_weights,
            standardize=standardize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        return super()._generate_params(X, y)

    def _generate_regularization(
        self,
        X: ArrayLike,
        beta: cp.Variable,
        parameters: SimpleNamespace,
        auxiliaries: Optional[SimpleNamespace] = None,
    ):
        group_regularization = AdaptiveGroupLasso._generate_regularization(
            self, X, beta, parameters, auxiliaries
        )
        # repetitive code...
        groups = np.arange(X.shape[1]) if self.groups is None else self.groups
        group_masks = [groups == i for i in np.sort(np.unique(groups))]
        ridge = cp.hstack([cp.sum_squares(beta[mask]) for mask in group_masks])
        ridge_regularization = 0.5 * parameters.delta @ ridge
        return group_regularization + ridge_regularization
