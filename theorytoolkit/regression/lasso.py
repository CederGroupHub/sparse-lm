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

from warnings import warn
import cvxpy as cp
import numpy as np
from theorytoolkit.regression.base import CVXEstimator

from .base import BaseEstimator


class Lasso(BaseEstimator):
    """
    Lasso Estimator implemented with cvxpy.
    """

    def __init__(self):
        super().__init__()

    def fit(self, feature_matrix, target_vector, sample_weight=None, \
            mu=None, log_mu_ranges=[(-3, 6)], log_mu_steps=[8]):
        """
        Fit the estimator. If mu not given, will optimize it.
        Inputs:
            feature_matrix(2d ArrayLike, n_structures*n_bit_orbits):
                Feature matrix of structures.
            target_vector(1d ArrayLike):
                Physical properties to fit.
            sample_weight(1d ArrayLike or None):
                Weight of samples. If not given, rows will be treated with equal weights.
            mu(1d arraylike of length 1 or None):
                mu parameter in LASSO regularization penalty term. Form is:
                L = ||Xw-y||^2 + mu * ||w||
                If None given, will be optimized.
                NOTE: You have to give mu as an array or list, because you have to match the form
                      in super().optimize_mu. Refer to the source for more detail.
            log_mu_ranges(None|List[(float,float)]):
                allowed optimization ranges of log(mu). If not provided, will be guessed.
                But I still highly recommend you to give this based on your experience.
            log_mu_steps(None|List[int]):
                Number of steps to search in each log_mu coordinate. Optional, but also
                recommeneded.
        Return:
            Optimized mu, cv score and coefficients.
            Fitter coefficients storeed in self.coef_.
        """
        if isinstance(mu, (int, float)):
            mu = [float(mu)]

        # Always call super().fit because this contains preprocessing of matrix
        # and vector, such as centering and weighting!
        if mu is None or len(mu) != 1:
            mu = super().optimize_mu(feature_matrix, target_vector,
                                     sample_weight=sample_weight,
                                     dim_mu=1,
                                     log_mu_ranges=log_mu_ranges,
                                     log_mu_steps=log_mu_steps)
            if mu[0] <= np.power(10, float(log_mu_ranges[0][0])):
                warnings.warn("Minimun allowed mu taken!")
            if mu[0] >= np.power(10, float(log_mu_ranges[0][1])):
                warnings.warn("Maximum allowed mu taken!")

        super().fit(feature_matrix, target_vector,
                    sample_weight=sample_weight,
                    mu=mu)
        return mu

    def _solve(self, feature_matrix, target_vector, mu=[0]):
        """
        X and y should already have been adjusted to account for weighting.
        mu(1D arraylike of length 1 or None):
           mu parameter in LASSO regularization penalty term. Form is:
           L = ||Xw-y|| + mu * ||w||
           If None given, will be optimized.
           I put mu as the last parameter, because in super().fit it is taken as
           part of *kwargs.
        """
        if mu[0] < 0:
            raise ValueError("Mu can not be negative!")

        A = feature_matrix.copy()
        b = target_vector.copy()
        n = A.shape[0]
        d = A.shape[1]

        w = cp.Variable((d,))
        z1 = cp.Variable((d,), pos=True)
        constraints = [z1 >= w, z1 >= -w]
        # Hierarchy constraints are not supported by regularization without L0

        # Cost function
        L = cp.sum_squares(A @ w - b) + mu[0] * cp.sum(z1)

        prob = cp.Problem(cp.Minimize(L), constraints)
        prob.solve()

        return w.value


class GroupLasso(CVXEstimator):
    """Group Lasso implementation.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * \sum_{G}||Beta_G||_2
    Where G represents groups of features/coeficients
    """

    # TODO set weights by sizes, inverse sizes etc here
    # TODO set groups by list of indices to allow overlap
    def __init__(self, groups, alpha=1.0, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
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
            **kwargs:
                Kewyard arguments passed to cvxpy solve.
                See docs linked in CVXEstimator base class for more info.
        """
        self.groups = np.asarray(groups)
        self.group_masks = [self.groups == i for i in np.unique(groups)]
        self.sizes = np.sqrt([sum(mask) for mask in self.group_masks])
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

    def _initialize_problem(self, X, y):
        super()._initialize_problem(X, y)
        grp_reg = cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + self._alpha * (self.sizes @ grp_reg)
        self._problem = cp.Problem(cp.Minimize(objective))


class SparseGroupLasso(GroupLasso):
    """Sparse Group Lasso.

    Regularized model:
        || X * Beta - y ||^2_2
            + alpha * l1_ratio * ||Beta||_1
            + alpha * (1 - l1_ratio) * \sum_{G}||Beta_G||_2
    Where G represents groups of features/coeficients
    """

    def __init__(self, groups, l1_ratio=0.5, alpha=1.0, fit_intercept=False,
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
                         fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

        if not 0 <= l1_ratio <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        elif l1_ratio == 0.0:
            warn(
                'It is more efficient to use GroupLasso directly than '
                'SparseGroupLasso with l1_ratio=0', UserWarning)
        elif l1_ratio == 1.0:
            warn(
                'It is more efficient to use Lasso directly than '
                'SparseGroupLasso with l1_ratio=1', UserWarning)

        self._lambda1 = cp.Parameter(nonneg=True, value=l1_ratio * alpha)
        self._lambda2 = cp.Parameter(nonneg=True, value=(1 - l1_ratio) * alpha)
        # save exact value so sklearn clone is happy dappy
        self._l1_ratio = l1_ratio

    @CVXEstimator.alpha.setter
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

    def _initialize_problem(self, X, y):
        super()._initialize_problem(X, y)
        l1_reg = cp.norm1(self._beta)
        grp_reg = cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + self._lambda1 * l1_reg + self._lambda2 * (self.sizes @ grp_reg)
        self._problem = cp.Problem(cp.Minimize(objective))


class AdaptiveLasso(CVXEstimator):
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

    def _initialize_problem(self, X, y):
        super()._initialize_problem(X, y)
        self._weights = cp.Parameter(shape=X.shape[1], nonneg=True,
                                     value=self.alpha * np.ones(X.shape[1]))
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + cp.norm1(cp.multiply(self._weights, self._beta))
        self._problem = cp.Problem(cp.Minimize(objective))

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


class AdaptiveGroupLasso(AdaptiveLasso, GroupLasso):
    """Adaptive Group Lasso, iteratively re-weighted group lasso.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * \sum_{G} w_G * ||Beta_G||_2

    Where w represents a vector of weights that is iteratively updated.
    """
    def __init__(self, groups, alpha=1.0, max_iter=5, eps=1E-6, tol=1E-10,
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
        super().__init__(groups=groups, alpha=alpha, max_iter=max_iter,
                         eps=eps, tol=tol, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

    def _initialize_problem(self, X, y):
        super()._initialize_problem(X, y)
        self._weights = cp.Parameter(shape=len(self.group_masks), nonneg=True,
                                     value=self.alpha * self.sizes)
        grp_reg = self._weights @ cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + grp_reg
        self._problem = cp.Problem(cp.Minimize(objective))

    def _update_weights(self, beta):
        self._previous_weights = self._weights.value
        group_norms = np.array(
            [np.linalg.norm(beta[mask]) for mask in self.group_masks])
        self._weights.value = (self.alpha * self.sizes) / (group_norms + self.eps)


# TODO allow adaptive weights on lasso/group/both
class AdaptiveSparseGroupLasso(AdaptiveLasso, SparseGroupLasso):
    """Adaptive Sparse Group Lasso, iteratively re-weighted sparse group lasso.

    Regularized model:
        || X * Beta - y ||^2_2
            + alpha * l1_ratio * ||w1^T Beta||_1
            + alpha * (1 - l1_ratio) * \sum_{G} w2_G * ||Beta_G||_2

    Where w1, w2 represent vectors of weights that is iteratively updated.
    """

    def __init__(self, groups,  l1_ratio=0.5, alpha=1.0, max_iter=5, eps=1E-6,
                 tol=1E-10, fit_intercept=False, normalize=False, copy_X=True,
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
                         max_iter=max_iter, eps=eps, tol=tol,
                         fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, warm_start=warm_start, solver=solver,
                         **kwargs)

    def _initialize_problem(self, X, y):
        super()._initialize_problem(X, y)
        self._weights = (
            cp.Parameter(shape=X.shape[1], nonneg=True,
                         value=self._lambda1.value * np.ones(X.shape[1])),
            cp.Parameter(shape=len(self.group_masks), nonneg=True,
                         value=self._lambda2.value * self.sizes),
            )
        l1_reg = cp.norm1(cp.multiply(self._weights[0], self._beta))
        grp_reg = self._weights[1] @ cp.hstack(
            [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + l1_reg + grp_reg
        self._problem = cp.Problem(cp.Minimize(objective))

    def _update_weights(self, beta):
        self._previous_weights = [self._weights[0].value,
                                  self._weights[1].value]
        self._weights[0].value = self._lambda1.value / (abs(beta) + self.eps)
        group_norms = np.array(
            [np.linalg.norm(beta[mask]) for mask in self.group_masks])
        self._weights[1].value = (self._lambda2.value * self.sizes) / (group_norms + self.eps)

    def _weights_converged(self):
        l1_converged = np.linalg.norm(
            self._weights[0].value - self._previous_weights[0]) <= self.tol
        group_converged = np.linalg.norm(
            self._weights[1].value - self._previous_weights[1]) <= self.tol
        return l1_converged and group_converged
