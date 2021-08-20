"""A set of generalized lasso estimators.
* Group Lasso
* Sparse Group Lasso
* Adaptive Lasso
* Adaptive Group Lasso
* Adaptive Sparse Group Lasso

Estimators follow scikit-learn interface, but use cvxpy to set up and solve
optimization problem.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

import warnings
import cvxpy as cp
import numpy as np
from theorytoolkit.regression.base import CVXEstimator


class Lasso(CVXEstimator):
    """
    Lasso Estimator implemented with cvxpy.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * ||Beta||_1
    Where w represents a vector of weights that is iteratively updated.
    """

    def __init__(self, alpha=1.0, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """
        Args:
            alpha (float):
                Regularization hyper-parameter.
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
        self._alpha = cp.Parameter(value=alpha, nonneg=True)
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, warm_start=warm_start, solver=solver,
                         **kwargs)

    @property
    def alpha(self):
        return self._alpha.value

    @alpha.setter
    def alpha(self, val):
        self._alpha.value = val

    def _gen_objective(self, X, y):
        # can also use cp.norm2(X @ self._beta - y)**2 not sure whats better
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
                    + self._alpha * cp.norm1(self._beta)
        return objective


class AdaptiveLasso(Lasso):
    """Adaptive Lasso implementation.

    Also known as iteratively re-weighted Lasso.
    Regularized model:
        || X * Beta - y ||^2_2 + alpha * ||w^T Beta||_1
    Where w represents a vector of weights that is iteratively updated.
    """

    # TODO allow different weight updates
    def __init__(self, alpha=1.0, max_iter=5, eps=1E-6, tol=1E-10,
                 fit_intercept=False, normalize=False, copy_X=True,
                 warm_start=False, solver=None, **kwargs):
        """Initialize estimator.

        Args:
            alpha (float):
                Regularization hyper-parameter.
            max_iter (int):
                Maximum number of re-weighting iteration steps.
            eps (float):
                Value to add to denominatar of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
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
                See docs linked in CVXEstimator base class for more info.
        """
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        self._weights, self._previous_weights = None, None

    def _gen_objective(self, X, y):
        self._weights = cp.Parameter(shape=X.shape[1], nonneg=True,
                                     value=self.alpha * np.ones(X.shape[1]))
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + cp.norm1(cp.multiply(self._weights, self._beta))
        return objective

    def _update_weights(self, beta):
        if beta is None and self._problem.value == -np.inf:
            raise RuntimeError(
                f"{self._problem} is infeasible.")
        self._previous_weights = self._weights.value
        self._weights.value = self.alpha / (abs(beta) + self.eps)

    def _weights_converged(self):
        return np.linalg.norm(
            self._weights.value - self._previous_weights) <= self.tol

    def _solve(self, X, y):
        problem = self._get_problem(X, y)
        problem.solve(solver=self.solver, warm_start=self.warm_start,
                      **self.solver_opts)
        for _ in range(self.max_iter - 1):
            self._update_weights(self._beta.value)
            problem.solve(solver=self.solver, warm_start=True,
                          **self.solver_opts)
            if self._weights_converged():
                break
        return self._beta.value


class GroupLasso(Lasso):
    """Group Lasso implementation.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * \sum_{G} w_G * ||Beta_G||_2
    Where G represents groups of features/coeficients
    """

    # TODO set groups by list of indices to allow overlap
    def __init__(self, groups, alpha=1.0, group_weights=None,
                 fit_intercept=False, normalize=False, copy_X=True,
                 warm_start=False, solver=None, **kwargs):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specefies the group
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
                See docs linked in CVXEstimator base class for more info.
        """
        self.groups = np.asarray(groups)
        self.group_masks = [self.groups == i for i in np.unique(groups)]

        if group_weights is not None:
            if len(group_weights) != len(self.group_masks):
                raise ValueError(
                    'group_weights must be the same length as the number of '
                    f'groups:  {len(group_weights)} != {len(self.group_masks)}')
            self.group_weights = np.array(group_weights)
        else:
            self.group_weights = np.sqrt(
                [sum(mask) for mask in self.group_masks])
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

    def _gen_objective(self, X, y):
        grp_reg = cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + self._alpha * (self.group_weights @ grp_reg)
        return objective


class SparseGroupLasso(GroupLasso):
    """Sparse Group Lasso.

    Regularized model:
        || X * Beta - y ||^2_2
            + alpha * l1_ratio * ||Beta||_1
            + alpha * (1 - l1_ratio) * \sum_{G}||Beta_G||_2
    Where G represents groups of features/coeficients
    """

    def __init__(self, groups, l1_ratio=0.5, alpha=1.0, group_weights=None,
                 fit_intercept=False, normalize=False, copy_X=True,
                 warm_start=False, solver=None, **kwargs):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specefies the group
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
                See docs linked in CVXEstimator base class for more info.
        """
        super().__init__(groups=groups, alpha=alpha,
                         group_weights=group_weights,
                         fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

        if not 0 <= l1_ratio <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        elif l1_ratio == 0.0:
            warnings.warn(
                'It is more efficient to use GroupLasso directly than '
                'SparseGroupLasso with l1_ratio=0', UserWarning)
        elif l1_ratio == 1.0:
            warnings.warn(
                'It is more efficient to use Lasso directly than '
                'SparseGroupLasso with l1_ratio=1', UserWarning)

        self._lambda1 = cp.Parameter(nonneg=True, value=l1_ratio * alpha)
        self._lambda2 = cp.Parameter(nonneg=True, value=(1 - l1_ratio) * alpha)
        # save exact value so sklearn clone is happy dappy
        self._l1_ratio = l1_ratio

    @Lasso.alpha.setter
    def alpha(self, val):
        self._alpha.value = val
        self._lambda1.value = self.l1_ratio * val
        self._lambda2.value = (1 - self.l1_ratio) * val

    @property
    def l1_ratio(self):
        return self._l1_ratio

    @l1_ratio.setter
    def l1_ratio(self, val):
        if not 0 <= val <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        self._l1_ratio = val
        self._lambda1.value = val * self.alpha
        self._lambda2.value = (1 - val) * self.alpha

    def _gen_objective(self, X, y):
        l1_reg = cp.norm1(self._beta)
        grp_reg = cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + self._lambda1 * l1_reg \
            + self._lambda2 * (self.group_weights @ grp_reg)
        return objective


class AdaptiveGroupLasso(AdaptiveLasso, GroupLasso):
    """Adaptive Group Lasso, iteratively re-weighted group lasso.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * \sum_{G} w_G * ||Beta_G||_2

    Where w represents a vector of weights that is iteratively updated.
    """
    def __init__(self, groups, alpha=1.0, group_weights=None, max_iter=5,
                 eps=1E-6, tol=1E-10, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specefies the group
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
                Value to add to denominatar of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
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
                See docs linked in CVXEstimator base class for more info.
        """
        # call with keywords to avoid MRO issues
        super().__init__(groups=groups, alpha=alpha,
                         group_weights=group_weights, max_iter=max_iter,
                         eps=eps, tol=tol, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

    def _gen_objective(self, X, y):
        self._weights = cp.Parameter(shape=len(self.group_masks), nonneg=True,
                                     value=self.alpha * self.group_weights)
        grp_reg = self._weights @ cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + grp_reg
        return objective

    def _update_weights(self, beta):
        self._previous_weights = self._weights.value
        group_norms = np.array(
            [np.linalg.norm(beta[mask]) for mask in self.group_masks])
        self._weights.value = (self.alpha * self.group_weights) / (group_norms + self.eps)


# TODO allow adaptive weights on lasso/group/both
class AdaptiveSparseGroupLasso(AdaptiveLasso, SparseGroupLasso):
    """Adaptive Sparse Group Lasso, iteratively re-weighted sparse group lasso.

    Regularized model:
        || X * Beta - y ||^2_2
            + alpha * l1_ratio * ||w1^T Beta||_1
            + alpha * (1 - l1_ratio) * \sum_{G} w2_G * ||Beta_G||_2

    Where w1, w2 represent vectors of weights that is iteratively updated.
    """

    def __init__(self, groups,  l1_ratio=0.5, alpha=1.0, group_weights=None,
                 max_iter=5, eps=1E-6, tol=1E-10, fit_intercept=False,
                 normalize=False, copy_X=True, warm_start=False, solver=None,
                 **kwargs):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specefies the group
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
                Value to add to denominatar of weights.
            tol (float):
                Absolute convergence tolerance for difference between weights
                at successive steps.
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
                See docs linked in CVXEstimator base class for more info.
        """
        # call with keywords to avoid MRO issues
        super().__init__(groups=groups, l1_ratio=l1_ratio, alpha=alpha,
                         group_weights=group_weights,
                         max_iter=max_iter, eps=eps, tol=tol,
                         fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, warm_start=warm_start, solver=solver,
                         **kwargs)

    def _gen_objective(self, X, y):
        self._weights = (
            cp.Parameter(shape=X.shape[1], nonneg=True,
                         value=self._lambda1.value * np.ones(X.shape[1])),
            cp.Parameter(shape=len(self.group_masks), nonneg=True,
                         value=self._lambda2.value * self.group_weights),
            )
        l1_reg = cp.norm1(cp.multiply(self._weights[0], self._beta))
        grp_reg = self._weights[1] @ cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + l1_reg + grp_reg
        return objective

    def _update_weights(self, beta):
        self._previous_weights = [self._weights[0].value,
                                  self._weights[1].value]
        self._weights[0].value = self._lambda1.value / (abs(beta) + self.eps)
        group_norms = np.array(
            [np.linalg.norm(beta[mask]) for mask in self.group_masks])
        self._weights[1].value = (self._lambda2.value * self.group_weights) / (group_norms + self.eps)

    def _weights_converged(self):
        l1_converged = np.linalg.norm(
            self._weights[0].value - self._previous_weights[0]) <= self.tol
        group_converged = np.linalg.norm(
            self._weights[1].value - self._previous_weights[1]) <= self.tol
        return l1_converged and group_converged
