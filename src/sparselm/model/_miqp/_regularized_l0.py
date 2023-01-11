"""MIQP based solvers for sparse solutions with hierarchical constraints.

Generalized regularized l0 solvers that allow grouping parameters as detailed in:

    https://doi.org/10.1287/opre.2015.1436

L1L0 proposed by Wenxuan Huang:

    https://arxiv.org/abs/1807.10753

L2L0 proposed by Peichen Zhong:

    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.024203

Estimators allow optional inclusion of hierarchical constraints at the single coefficient
or group of coefficients level.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"


import warnings
from abc import ABCMeta, abstractmethod

import cvxpy as cp

from ._base import MIQP_L0
from sparselm.model._base import TikhonovMixin


class RegularizedL0(MIQP_L0):
    """Implementation of mixed-integer quadratic programming l0 regularized estimator.

    Supports grouping parameters and group-level hierarchy, but requires groups as a
    compulsory argument.
    """

    def __init__(
        self,
        groups,
        alpha=1.0,
        big_M=1000,
        hierarchy=None,
        ignore_psd_check=True,
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
                each parameter corresponds to. One dimensional.
                Note: in cluster expansion, a group can either be each correlation
                function (function-level hierarchy) or be each cluster orbit
                (orbit-level hierarchy). In the first case, each index should be its
                own group, so use range(len(ecis)). In the second case, correlation
                functions under the same orbit will be considered as the same group,
                so use function_orbit_ids here.
                Each external term in cluster subspace should be considered as its
                own group.
            alpha (float):
                Regularization hyper-parameter.
            big_M (float):
                Upper bound on the norm of coefficients associated with each
                cluster (groups of coefficients) ||Beta_c||_2
            hierarchy (list):
                A list of lists of integers storing hierarchy relations between
                groups.
                Each sublist contains indices of other groups
                on which the group associated with each element of
                the list depends. i.e. hierarchy = [[1, 2], [0], []] mean that
                group 0 depends on 1, and 2; 1 depends on 0, and 2 has no
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
                cvxpy backend solver to use. Supported solvers are:
                ECOS, ECOS_BB, CVXOPT, SCS, GUROBI, Elemental.
                GLPK and GLPK_MI (via CVXOPT GLPK interface)
            solver_options (dict):
                dictionary of keyword arguments passed to cvxpy solve.
                See docs in CVXEstimator for more information.
        """
        super().__init__(
            groups=groups,
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

        self._alpha = alpha
        self._lambda0 = cp.Parameter(nonneg=True, value=alpha)

    @property
    def alpha(self):
        """Get alpha hyperparameter value."""
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        """Set alpha hyperparameter value."""
        self._alpha = val
        self._lambda0.value = val

    def _gen_objective(self, X, y):
        """Generate the quadratic form and l0 regularization portion of objective."""
        c0 = 2 * X.shape[0]  # keeps hyperparameter scale independent
        objective = super()._gen_objective(X, y) + c0 * self._lambda0 * cp.sum(self._z0)
        return objective


class MixedL0(RegularizedL0, metaclass=ABCMeta):
    """Abstract base class for mixed L0 regularization models: L1L0 and L2L0."""

    def __init__(
        self,
        groups,
        alpha=1.0,
        l0_ratio=0.5,
        big_M=1000,
        hierarchy=None,
        ignore_psd_check=True,
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
                each parameter corresponds to. One dimensional.
                Note: in cluster expansion, a group can either be each correlation
                function (function-level hierarchy) or be each cluster orbit
                (orbit-level hierarchy). In the first case, each index should be its
                own group, so use range(len(ecis)). In the second case, correlation
                functions under the same orbit will be considered as the same group,
                so use function_orbit_ids here.
                Each external term in cluster subspace should be considered as its
                own group.
            alpha (float):
                Regularization hyper-parameter.
            l0_ratio (float):
                Mixing parameter between l1 and l0 regularization.
            big_M (float):
                Upper bound on the norm of coefficients associated with each
                cluster (groups of coefficients) ||Beta_c||_2
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
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )

        if not 0 <= l0_ratio <= 1:
            raise ValueError("l0_ratio must be between 0 and 1.")
        elif l0_ratio == 0.0:
            warnings.warn(
                "It's more efficient to use Ridge/Lasso instead of l0_ratio=0",
                UserWarning,
            )

        self._lambda0.value = l0_ratio * alpha
        self._lambda2 = cp.Parameter(nonneg=True, value=(1 - l0_ratio) * alpha)
        #  save exact value so sklearn clone is happy dappy
        self._l0_ratio = l0_ratio

    @RegularizedL0.alpha.setter
    def alpha(self, val):
        """Set hyperparameter values."""
        self._alpha = val
        self._lambda0.value = self.l0_ratio * val
        self._lambda2.value = (1 - self.l0_ratio) * val

    @property
    def l0_ratio(self):
        """Get l0 ratio."""
        return self._l0_ratio

    @l0_ratio.setter
    def l0_ratio(self, val):
        """Set l0 ratio."""
        if not 0 <= val <= 1:
            raise ValueError("l0_ratio must be between 0 and 1.")
        self._l0_ratio = val
        self._lambda0.value = val * self.alpha
        self._lambda2.value = (1 - val) * self.alpha

    @abstractmethod
    def _gen_objective(self, X, y):
        """Generate optimization objective."""
        return super()._gen_objective(X, y)


class L1L0(MixedL0):
    """L1L0 regularized estimator.

    Estimator with L1L0 regularization solved with mixed integer programming
    as discussed in:

    https://arxiv.org/abs/1807.10753

    Installation of Gurobi is not a must, but highly recommended.
    You can get a free academic gurobi license...
    ECOS_BB also works but can be very slow.

    Regularized model is:

    .. math::

        ||X \beta - y||^2 + \alpha * l0_ratio * ||\beta||_0
                           + \alpha * (1 - l0_ratio) * ||\beta||_1
    """

    def __init__(
        self,
        groups,
        alpha=1.0,
        l0_ratio=0.5,
        big_M=1000,
        hierarchy=None,
        ignore_psd_check=True,
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
                each parameter corresponds to. One dimensional.
                Note: in cluster expansion, a group can either be each correlation
                function (function-level hierarchy) or be each cluster orbit
                (orbit-level hierarchy). In the first case, each index should be its
                own group, so use range(len(ecis)). In the second case, correlation
                functions under the same orbit will be considered as the same group,
                so use function_orbit_ids here.
                Each external term in cluster subspace should be considered as its
                own group.
            alpha (float):
                Regularization hyper-parameter.
            l0_ratio (float):
                Mixing parameter between l1 and l0 regularization.
            big_M (float):
                Upper bound on the norm of coefficients associated with each
                cluster (groups of coefficients) ||Beta_c||_2
            hierarchy (list):
                A list of lists of integers storing hierarchy relations between
                coefficients.
                Each sublist contains indices of other coefficients
                on which the coefficient associated with each element of
                the list depends. i.e. hierarchy = [[1, 2], [0], []] mean that
                coefficient 0 depends on 1, and 2; 1 depends on 0, and 2 has no
                dependence.
            ignore_psd_check (bool):
                Whether to ignore cvxpy's PSD checks of matrix used in quadratic
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
            l0_ratio=l0_ratio,
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )
        self._z1 = None

    def _gen_constraints(self, X, y):
        """Generate the constraints used to solve l1l0 regularization."""
        constraints = super()._gen_constraints(X, y)
        # L1 constraints (why not do an l1 norm in the objective instead?)
        constraints += [self._z1 >= self._beta, self._z1 >= -1.0 * self._beta]
        return constraints

    def _gen_objective(self, X, y):
        """Generate the objective function used in l1l0 regression model."""
        self._z1 = cp.Variable(X.shape[1])
        c0 = 2 * X.shape[0]  # keeps hyperparameter scale independent
        objective = super()._gen_objective(X, y) + c0 * self._lambda2 * cp.sum(self._z1)

        return objective


class L2L0(TikhonovMixin, MixedL0):
    """L2L0 regularized estimator.

    Based on estimator with L2L0 regularization solved with mixed integer programming
    proposed by Peichen Zhong:

    https://arxiv.org/abs/2204.13789

    Extended to allow grouping of coefficients and group-level hierarchy as described
    in:

    https://doi.org/10.1287/opre.2015.1436

    And allows using a Tihkonov matrix in the l2 term.

    Regularized model is:

    .. math::

        ||X \beta - y||^2 + \alpha * l0_ratio * ||\beta||_0
                           + \alpha * (1 - l0_ratio) * ||\beta||^2_2
    """

    def __init__(
        self,
        groups,
        alpha=1.0,
        l0_ratio=0.5,
        big_M=1000,
        hierarchy=None,
        tikhonov_w=None,
        ignore_psd_check=True,
        fit_intercept=False,
        copy_X=True,
        warm_start=False,
        solver=None,
        solver_options=None,
    ):
        """Initialize L2L0 estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to. One dimensional.
                Note: in cluster expansion, a group can either be each correlation
                function (function-level hierarchy) or be each cluster orbit
                (orbit-level hierarchy). In the first case, each index should be its
                own group, so use range(len(ecis)). In the second case, correlation
                functions under the same orbit will be considered as the same group,
                so use function_orbit_ids here.
                Each external term in cluster subspace should be considered as its
                own group.
            alpha (float):
                Regularization hyper-parameter.
            l0_ratio (float):
                Mixing parameter between l1 and l0 regularization.
            big_M (float):
                Upper bound on the norm of coefficients associated with each
                cluster (groups of coefficients) ||Beta_c||_2
            hierarchy (list):
                A list of lists of integers storing hierarchy relations between
                coefficients.
                Each sublist contains indices of other coefficients
                on which the coefficient associated with each element of
                the list depends. i.e. hierarchy = [[1, 2], [0], []] mean that
                coefficient 0 depends on 1, and 2; 1 depends on 0, and 2 has no
                dependence.
            tikhonov_w (np.array):
                Matrix to add weights to L2 regularization.
            ignore_psd_check (bool):
                Wether to ignore cvxpy's PSD checks of matrix used in quadratic
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
            l0_ratio=l0_ratio,
            big_M=big_M,
            hierarchy=hierarchy,
            ignore_psd_check=ignore_psd_check,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            warm_start=warm_start,
            solver=solver,
            solver_options=solver_options,
        )
        self.tikhonov_w = tikhonov_w
