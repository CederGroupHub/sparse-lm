""" L1L0 regularization least square solver. """

__author__ = "Fengyu Xie, Luis Barroso-Luque"

import warnings
import cvxpy as cp
from .base import CVXEstimator


class L1L0(CVXEstimator):
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
        ||X * Beta - y||^2 + alpha * (1 - l1_ratio) * ||Beta||_0
                           + alpha * l1_ratio * ||Beta||_1
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, big_M=100, hierarchy=None,
                 **kwargs):
        """
        Args:
            l1_ratio (float):
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
            **kwargs:
                Keyword arguments to pass to CVXEstimator base class. See
                CVXEstimator docstring for more information.
        """
        super().__init__(**kwargs)
        # TODO should be check the solver that is chosen and raise a warning?

        if not 0 <= l1_ratio <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        elif l1_ratio == 1.0:
            warnings.warn(
                'It is more efficient to use Lasso instead of '
                'L1L0 with l1_ratio=1', UserWarning)

        self.hierarchy = hierarchy
        self._alpha = alpha
        self._big_M = cp.Parameter(nonneg=True, value=big_M)
        self._lambda1 = cp.Parameter(nonneg=True, value=l1_ratio * alpha)
        self._lambda0 = cp.Parameter(nonneg=True, value=(1 - l1_ratio) * alpha)
        # save exact value so sklearn clone is happy dappy
        self._l1_ratio = l1_ratio
        self._z0, self._z1 = None, None

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        self._alpha = val
        self._lambda1.value = self.l1_ratio * val
        self._lambda0.value = (1 - self.l1_ratio) * val

    @property
    def big_M(self):
        return self._big_M.value

    @big_M.setter
    def big_M(self, val):
        self._big_M.value = val

    @property
    def l1_ratio(self):
        return self._l1_ratio

    @l1_ratio.setter
    def l1_ratio(self, val):
        if not 0 <= val <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        self._l1_ratio = val
        self._lambda1.value = val * self.alpha
        self._lambda0.value = (1 - val) * self.alpha

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
        # TODO check if we have use to normalize by problem size here?
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + self._lambda0 * cp.sum(self._z0) \
            + self._lambda1 * cp.sum(self._z1)
        return objective

