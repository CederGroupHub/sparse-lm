""" L1L0 regularization least square solver. """

__author__ = "Fengyu Xie, Luis Barroso-Luque"

# TODO s
#  Should be check the solver that is chosen and raise a warning?
#  Is it better to normalize by problem size here 1/(2*X.shape[0])?
#  I changed user input to use hyperparameters alpha and l0_ratio to better match
#    other estimator signatures...
#  I removed the timeouts, are those necessary? or just add to docstrings so users know
#  Are these actually correctly implemented!?!?


import warnings
from abc import ABCMeta
import cvxpy as cp
from .base import CVXEstimator


class mixedL0(CVXEstimator, metaclass=ABCMeta):
    """Abstract base class for mixed L0 regularization models: L1L0
    
    Only defines the shared variables...
    """
    def __init__(self, alpha=1.0, l0_ratio=0.5, big_M=100, hierarchy=None,
                 fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """
        Args:
            alpha (float):
                Regularization hyper-parameter.
            l0_ratio (float):
                Mixing parameter between l1 and l0 regularization.
            big_M (float):
                Upper bound on the norm of coefficients associated with each
                cluster (groups of coefficients) ||Beta_c||_2
            hierarchy (2D array like):
                A list of integers storing hierarchy relations between
                coefficients.
                Each sublist contains indices of other coefficients
                that depend on the coefficient associated with each element of
                the list.
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
        
        if not 0 <= l0_ratio <= 1:
            raise ValueError('l0_ratio must be between 0 and 1.')
        elif l0_ratio == 0.0:
            warnings.warn(
                'It is more efficient to use Lasso instead of l0_ratio=1',
                UserWarning)

        self.hierarchy = hierarchy
        self._alpha = alpha
        self._big_M = cp.Parameter(nonneg=True, value=big_M)
        self._lambda0 = cp.Parameter(nonneg=True, value=l0_ratio * alpha)
        self._lambda1 = cp.Parameter(nonneg=True, value=(1 - l0_ratio) * alpha)
        # save exact value so sklearn clone is happy dappy
        self._l0_ratio = l0_ratio
        self._z0 = None

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val
        self._lambda0.value = self.l0_ratio * val
        self._lambda1.value = (1 - self.l0_ratio) * val

    @property
    def big_M(self):
        return self._big_M.value

    @big_M.setter
    def big_M(self, val):
        self._big_M.value = val

    @property
    def l0_ratio(self):
        return self._l0_ratio

    @l0_ratio.setter
    def l0_ratio(self, val):
        if not 0 <= val <= 1:
            raise ValueError('l0_ratio must be between 0 and 1.')
        self._l0_ratio = val
        self._lambda1.value = val * self.alpha
        self._lambda0.value = (1 - val) * self.alpha


class L1L0(mixedL0):
    """
    Estimator with L1L0 regularization solved with mixed integer programming
    as discussed in:
    https://arxiv.org/abs/1807.10753

    Removed Gurobi dependency!
    Uses any solver it has in hand. If nothing, will use ECOS_BB.
    ECOS_BB is only supported after cvxpy-1.1.6. It will not be given
    automatically pulled up by cvxpy, and has to be explicitly called
    passed in the constructor with solver='ECOS_BB'.

    Installation of Gurobi is no longer a must, but highly recommended,
    since ECOS_BB can be very slow. You may also choose CPLEX, etc.

    Regularized model is:
        ||X * Beta - y||^2 + alpha * (1 - l0_ratio) * ||Beta||_0
                           + alpha * l0_ratio * ||Beta||_1
    """
    def __init__(self, alpha=1.0, l0_ratio=0.5, big_M=100, hierarchy=None,
                 fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        super().__init__(alpha=alpha, l0_ratio=l0_ratio, big_M=big_M,
                         hierarchy=hierarchy, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)
        self._z1 = None

    def _gen_constraints(self, X, y):
        """Generate the constraints used to solve l1l0 regularization"""

        self._z0 = cp.Variable(X.shape[1], integer=True)
        self._z1 = cp.Variable(X.shape[1], pos=True)
        constraints = [0 <= self._z0,
                       self._z0 <= 1,
                       self._big_M * self._z0 >= self._beta,
                       self._big_M * self._z0 >= -self._beta,
                       self._z1 >= self._beta,
                       self._z1 >= -self._beta]

        # Hierarchy constraints.
        if self.hierarchy is not None:
            for sub_id, high_ids in enumerate(self.hierarchy):
                for high_id in high_ids:
                    constraints.append(self._z0[high_id] <= self._z0[sub_id])

    def _gen_objective(self, X, y):
        """Generate the objective function used in l1l0 regression model"""
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + self._lambda0 * cp.sum(self._z0) \
            + self._lambda1 * cp.sum(self._z1)
        return objective


class L2L0(mixedL0):
    """
    Estimator with L1L0 regularization solved with mixed integer programming
    proposed by Peichen Zhong.

    Regularized model is:
        ||X * Beta - y||^2 + alpha * (1 - l0_ratio) * ||Beta||_0
                           + alpha * l0_ratio * ||Beta||_2
    """

    def _gen_constraints(self, X, y):
        """Generate the constraints used to solve l1l2 regularization"""

        self._z0 = cp.Variable(X.shape[1], integer=True)
        constraints = [0 <= self._z0,
                       self._z0 <= 1,
                       self._big_M * self._z0 >= self._beta,
                       self._big_M * self._z0 >= -self._beta]

        # Hierarchy constraints.
        if self.hierarchy is not None:
            for sub_id, high_ids in enumerate(self.hierarchy):
                for high_id in high_ids:
                    constraints.append(self._z0[high_id] <= self._z0[sub_id])

    def _gen_objective(self, X, y):
        """Generate the objective function used in l1l0 regression model"""
        
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + self._lambda0 * cp.sum(self._z0) \
            + self._lambda1 * cp.sum_squares(self._beta)
        return objective
