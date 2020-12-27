"""A set of generalized lasso estimators.

Estimators follow scikit-learn interface, but use cvxpy to set up and solver
optimization problem.
"""

import cvxpy as cp
import numpy as np
from theorytoolkit.regression.base import CVXEstimator

__author__ = "Luis Barroso-Luque"


class AdaptiveLasso(CVXEstimator):
    """Adaptive Lasso implementation.

    Also known as iteratively re-weighted Lasso.
    Regularized model:

    """
    def __init__(self, alpha=1.0, max_iter=5, eps=1E-10, tol=1E-10,
                 fit_intercept=False, normalize=False, copy_X=True,
                 warm_start=False, solver=None, verbose=False, **kwargs):
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
            verbose (bool):
                Print cvxpy solver messages.
            **kwargs:
                Kewyard arguments passed to cvxpy solve.
                See docs linked in CVXEstimator base class for more info.
        """
        self._alpha = cp.Parameter(value=alpha, nonneg=True)
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        super().__init__(fit_intercept, normalize, copy_X, warm_start, solver,
                         verbose, **kwargs)

    @property
    def alpha(self):
        return self._alpha.value

    @alpha.setter
    def alpha(self, val):
        self._alpha.value = val

    def _initialize_problem(self, X, y):
        self._X = X
        self._y = y
        self._beta = cp.Variable(shape=X.shape[1])
        self._w = cp.Parameter(shape=X.shape[1], nonneg=True,
                               value=np.ones(X.shape[1]))
        objective = cp.sum_squares(X @ self._beta - y) \
            + self.alpha * cp.norm1(cp.multiply(self._w, self._beta))
        self._problem = cp.Problem(cp.Minimize(objective))

    def _solve(self, X, y):
        problem = self._get_problem(X, y)
        problem.solve(warm_start=self.warm_start)
        w_prev = self._w.value
        for _ in range(self.max_iter - 1):
            self._w.value = 1.0 / (abs(self._beta.value) + self.eps)
            problem.solve(warm_start=self.warm_start)
            if np.linalg.norm(self._w.value - w_prev) <= self.tol:
                break
            w_prev = self._w.value
        return self._beta.value


class GroupLasso(CVXEstimator):
    """Group Lasso implementation.

    Regularized model:

    """

    def __init__(self, groups, alpha=1.0, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, verbose=False,
                 **kwargs):
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
            verbose (bool):
                Print cvxpy solver messages.
            **kwargs:
                Kewyard arguments passed to cvxpy solve.
                See docs linked in CVXEstimator base class for more info.
        """
        self.groups = np.asarray(groups)
        self._alpha = cp.Parameter(value=alpha, nonneg=True)
        super().__init__(fit_intercept, normalize, copy_X, warm_start, solver,
                         verbose, **kwargs)

    @property
    def alpha(self):
        return self._alpha.value

    @alpha.setter
    def alpha(self, val):
        self._alpha.value = val

    def _initialize_problem(self, X, y):  # TODO set weights by sizes etc here
        self._X = X
        self._y = y
        group_inds = np.unique(self.groups)
        sizes = np.sqrt([sum(self.groups == i) for i in group_inds])
        self._beta = cp.Variable(X.shape[1])
        grp_reg = cp.hstack(
            [cp.norm2(self._beta[self.groups == i]) for i in group_inds])
        objective = cp.sum_squares(X @ self._beta - y) \
            + self.alpha * (sizes @ grp_reg)
        self._problem = cp.Problem(cp.Minimize(objective))


class SparseGroupLasso(CVXEstimator):
    """Sparse Group Lasso.

    Regularized model:

    """

    def __init__(self, groups, l1_ratio=0.5, alpha=1.0, fit_intercept=False,
                 normalize=False, copy_X=True, warm_start=False, solver=None,
                 verbose=False, **kwargs):
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
            verbose (bool):
                Print cvxpy solver messages.
            **kwargs:
                Kewyard arguments passed to cvxpy solve.
                See docs linked in CVXEstimator base class for more info.
        """
        self.groups = np.asarray(groups)
        self._alpha = cp.Parameter(value=alpha, nonneg=True)
        self._lambda1 = cp.Parameter(nonneg=True)
        self._lambda2 = cp.Parameter(nonneg=True)
        self.l1_ratio = l1_ratio
        super().__init__(fit_intercept, normalize, copy_X, warm_start, solver,
                         verbose, **kwargs)

    @property
    def alpha(self):
        return self._alpha.value

    @alpha.setter
    def alpha(self, val):
        self._alpha.value = val
        self.l1_ratio = self.l1_ratio

    @property
    def l1_ratio(self):
        if self._lambda1.value == 0:
            return 0.0
        return 1.0 - 0.5 * self._lambda2.value/self._lambda1.value

    @l1_ratio.setter
    def l1_ratio(self, val):
        if not 0 <= val <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        self._lambda1.value = val * self.alpha
        self._lambda2.value = (1 - val) * self.alpha

    def _initialize_problem(self, X, y):  # TODO set weights by sizes etc here
        group_inds = np.unique(self.groups)
        sizes = np.sqrt([sum(self.groups == i) for i in group_inds])
        self._beta = cp.Variable(X.shape[1])
        self._X = X
        self._y = y
        l1_reg = cp.norm1(self._beta)
        grp_reg = cp.hstack(
            [cp.norm2(self._beta[self.groups == i]) for i in group_inds])
        objective = cp.sum_squares(X @ self._beta - y) \
            + self._lambda1 * l1_reg + self._lambda2 * (sizes @ grp_reg)
        self._problem = cp.Problem(cp.Minimize(objective))
