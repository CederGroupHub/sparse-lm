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
        max_iter=5,
        eps=1e-6,
        tol=1e-10,
        update_function=None,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
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
        self._weights, self._previous_weights = None, None

    def _validate_params(self, X: ArrayLike, y: ArrayLike):
        super()._validate_params(X, y)
        check_scalar(self.max_iter, "max_iter", int, min_val=1)
        check_scalar(self.eps, "eps", float)
        check_scalar(self.tol, "tol", float)

        if self.max_iter == 1:
            warnings.warn(
                "max_iter is set to 1. It should ideally be set > 1, otherwise reconsider "
                "using a non-adaptive estimator",
                UserWarning,
            )

        if self.update_function is None:
            self.update_function = lambda beta, eps: 1.0 / (abs(beta) + eps)

    def _gen_regularization(self, X):
        self._weights = cp.Parameter(
            shape=X.shape[1], nonneg=True, value=self.alpha * np.ones(X.shape[1])
        )
        return cp.norm1(cp.multiply(self._weights, self.beta_))

    def _update_weights(self, beta):
        if beta is None and self.problem.value == -np.inf:
            raise RuntimeError(f"{self.problem} is infeasible.")
        self._previous_weights = self._weights.value
        self._weights.value = self.alpha * self.update_function(abs(beta), self.eps)

    def _weights_converged(self):
        return np.linalg.norm(self._weights.value - self._previous_weights) <= self.tol

    def _solve(self, X, y, *args, **kwargs):
        problem = self._get_problem(X, y)
        problem.solve(
            solver=self.solver, warm_start=self.warm_start, **self.solver_options
        )
        for _ in range(self.max_iter - 1):
            self._update_weights(self.beta_.value)
            problem.solve(solver=self.solver, warm_start=True, **self.solver_options)
            if self._weights_converged():
                break
        return self.beta_.value


class AdaptiveGroupLasso(AdaptiveLasso, GroupLasso):
    r"""Adaptive Group Lasso, iteratively re-weighted group lasso.

    Regularized model:

    .. math::

        || X \beta - y ||^2_2 + \alpha * \sum_{G} w_G ||\beta_G||_2

    Where w represents a vector of weights that is iteratively updated.
    """

    def __init__(
        self,
        groups,
        alpha=1.0,
        group_weights=None,
        max_iter=5,
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

    def _gen_regularization(self, X):
        grp_norms = self._gen_group_norms(X)
        self._weights = cp.Parameter(
            shape=len(self._group_masks),
            nonneg=True,
            value=self.alpha * self.group_weights,
        )
        return self._weights @ grp_norms

    def _update_weights(self, beta):
        self._previous_weights = self._weights.value
        self._weights.value = (self.alpha * self.group_weights) * self.update_function(
            self._group_norms.value, self.eps
        )


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
        group_list,
        alpha=1.0,
        group_weights=None,
        max_iter=5,
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

    def _validate_params(self, X, y):
        # call directly since this has a diamond inheritance
        check_scalar(self.max_iter, "max_iter", int, min_val=1)
        check_scalar(self.eps, "eps", float)
        check_scalar(self.tol, "tol", float)

        if self.max_iter == 1:
            warnings.warn(
                "max_iter is set to 1. It should ideally be set > 1, otherwise reconsider "
                "using a non-adaptive estimator",
                UserWarning,
            )

        if self.update_function is None:
            self.update_function = lambda beta, eps: 1.0 / (abs(beta) + eps)

        OverlapGroupLasso._validate_params(self, X, y)

    def _gen_objective(self, X, y):
        return AdaptiveGroupLasso._gen_objective(self, X, y)

    def _solve(self, X, y, *args, **kwargs):
        beta = AdaptiveGroupLasso._solve(
            self, X[:, self.beta_indices], y, *args, **kwargs
        )
        beta = np.array([sum(beta[self.beta_indices == i]) for i in range(X.shape[1])])
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
        max_iter=5,
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

    def _gen_regularization(self, X):
        grp_norms = self._gen_group_norms(X)
        self._weights = (
            cp.Parameter(
                shape=X.shape[1],
                nonneg=True,
                value=self._lambda1.value * np.ones(X.shape[1]),
            ),
            cp.Parameter(
                shape=(len(self._group_masks),),
                nonneg=True,
                value=self._lambda2.value * self.group_weights,
            ),
        )
        l1_reg = cp.norm1(cp.multiply(self._weights[0], self.beta_))
        grp_reg = self._weights[1] @ grp_norms
        return l1_reg + grp_reg

    def _update_weights(self, beta):
        self._previous_weights = [self._weights[0].value, self._weights[1].value]
        self._weights[0].value = self._lambda1.value / (abs(beta) + self.eps)
        self._weights[1].value = (
            self._lambda2.value * self.group_weights
        ) * self.update_function(self._group_norms.value, self.eps)

    def _weights_converged(self):
        l1_converged = (
            np.linalg.norm(self._weights[0].value - self._previous_weights[0])
            <= self.tol
        )
        group_converged = (
            np.linalg.norm(self._weights[1].value - self._previous_weights[1])
            <= self.tol
        )
        return l1_converged and group_converged


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
        delta=1.0,
        group_weights=None,
        max_iter=5,
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

    def _gen_group_norms(self, X):
        return RidgedGroupLasso._gen_group_norms(self, X)

    def _gen_regularization(self, X):
        reg = AdaptiveGroupLasso._gen_regularization(self, X)
        ridge = cp.hstack(
            [cp.sum_squares(self.beta_[mask]) for mask in self._group_masks]
        )
        return reg + 0.5 * self._delta @ ridge
