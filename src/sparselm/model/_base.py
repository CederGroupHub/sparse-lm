"""Base classes for in-house linear regression estimators.

The classes make use of and follow the scikit-learn API.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from abc import ABCMeta, abstractmethod

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import (
    LinearModel,
    _check_sample_weight,
    _preprocess_data,
    _rescale_data,
)


class CVXEstimator(RegressorMixin, LinearModel, metaclass=ABCMeta):
    """Abstract base class for estimators using cvxpy with a sklearn interface.

    Note cvxpy can use one of many 3rd party solvers, default is most often
    CVXOPT. The solver can be specified by setting the solver keyword argument.
    And can solver specific settings can be set by passing a dictionary of
    solver_options.

    See "Setting solver options" in documentation for details of available options:
    https://www.cvxpy.org/tutorial/advanced/index.html#advanced

    Keyword arguments are the same as those found in sklearn linear models.
    """

    def __init__(
        self,
        fit_intercept: bool = False,
        copy_X: bool = True,
        warm_start: bool = False,
        solver: str = None,
        solver_options: dict = None,
    ):
        """Initialize estimator.

        Args:
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
                See docs linked above for more information.
        """
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.solver = solver

        if solver_options is None:
            self.solver_options = {}
        else:
            self.solver_options = solver_options

        self._problem, self._beta, self._X, self._y = None, None, None, None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        *args,
        **kwargs
    ):
        """Prepare fit input with sklearn help then call fit method.

        Args:
            X (ArrayLike):
                Training data of shape (n_samples, n_features).
            y (ArrayLike):
                Target values. Will be cast to X's dtype if necessary
                of shape (n_samples,) or (n_samples, n_targets)
            sample_weight (ArrayLike):
                Individual weights for each sample of shape (n_samples,)
                default=None
            *args:
                Positional arguments passed to _fit method
            **kwargs:
                Keyword arguments passed to _fit method

        Returns:
            instance of self
        """
        X, y = self._validate_data(
            X, y, accept_sparse=False, y_numeric=True, multi_output=True
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            copy=self.copy_X,
            fit_intercept=self.fit_intercept,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            X, y, _ = _rescale_data(X, y, sample_weight)

        self._validate_params(X, y)
        self.coef_ = self._solve(X, y, *args, **kwargs)
        self._set_intercept(X_offset, y_offset, X_scale)

        # return self for chaining fit and predict calls
        return self

    def _validate_params(self, X: ArrayLike, y: ArrayLike):
        """Validate hyper parameters.

        Implement this in an estimator to check for valid hyper parameters.
        """
        return

    @abstractmethod
    def _gen_objective(self, X: ArrayLike, y: ArrayLike):
        """Define the cvxpy objective function represeting regression model.

        The objective must be stated for a minimization problem.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector

        Returns:
            cvpx Expression
        """
        return None

    def _gen_constraints(self, X: ArrayLike, y: ArrayLike):
        """Generate constraints for optimization problem.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector

        Returns:
            list of cvpx constraints
        """
        return None

    def _initialize_problem(self, X: ArrayLike, y: ArrayLike):
        """Initialize cvxpy problem from the generated objective function.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector
        """
        self._beta = cp.Variable(X.shape[1])
        self._X = X
        self._y = y
        objective = self._gen_objective(X, y)
        constraints = self._gen_constraints(X, y)
        self._problem = cp.Problem(cp.Minimize(objective), constraints)

    def _get_problem(self, X: ArrayLike, y: ArrayLike):
        """Define and create cvxpy optimization problem."""
        if self._problem is None:
            self._initialize_problem(X, y)
        elif not np.array_equal(X, self._X) or not np.array_equal(y, self._y):
            self._initialize_problem(X, y)
        return self._problem

    def _solve(self, X: ArrayLike, y: ArrayLike, *args, **kwargs):
        """Solve the cvxpy problem."""
        problem = self._get_problem(X, y)
        problem.solve(
            solver=self.solver, warm_start=self.warm_start, **self.solver_options
        )
        return self._beta.value


class TikhonovMixin:
    """Mixin class to add a Tihhonov/ridge regularization term.

    When using this Mixin, a cvxpy parameter should be set as the _eta attribute
    and an attribute tikhonov_w can be added to allow a matrix otherwise simple l2/Ridge
    is used.
    """

    def _gen_objective(self, X, y):
        """Add a Tikhnonov regularization term to the objective function."""
        c0 = 2 * X.shape[0]  # keeps hyperparameter scale independent

        if hasattr(self, "tikhonov_w") and self.tikhonov_w is not None:
            tikhonov_w = self.tikhonov_w
        else:
            tikhonov_w = np.eye(X.shape[1])

        objective = super()._gen_objective(X, y) + c0 * self._eta * cp.sum_squares(
            tikhonov_w @ self._beta
        )

        return objective
