"""Base classes for in-house linear regression estimators.

The classes make use of and follow the scikit-learn API.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from abc import ABCMeta, abstractmethod
from types import SimpleNamespace
from typing import NamedTuple, Optional

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


class CVXCanonicals(NamedTuple):
    """CVXpy Canonical objects representing the underlying optimization problem.

    Attributes:
        objective (cp.Problem):
            Objective function.
        objective (cp.Expression):
            Objective function.
        beta (cp.Variable):
            Variable to be optimized (corresponds to the estimated coef_ attribute).
        auxiliaries (SimpleNamespace of cp.Variable or cp.Expression):
            SimpleNamespace with auxiliary cp.Variable or cp.Expression objects.
            The namespace should be defined by the estimator generating it.
        parameters (SimpleNamespace of cp.Parameter):
            SimpleNamespace with named cp.Parameter objects. The namespace should be defined
            by the estimator generating it.
        constraints (list of cp.Constaint):
            List of constraints.
    """

    problem: cp.Problem
    objective: cp.Expression
    beta: cp.Variable
    auxiliaries: Optional[SimpleNamespace]
    parameters: Optional[SimpleNamespace]
    constraints: Optional[list[cp.Expression]]


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
        self.solver_options = solver_options

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
            X, y, accept_sparse=False, y_numeric=True, multi_output=False
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
            # rescale sample_weight to sum to number of samples
            sample_weight = sample_weight * (X.shape[0] / np.sum(sample_weight))

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

        if not hasattr(self, "canonicals_") or self.warm_start is False:
            self._initialize_problem(X, y)

        if self.warm_start is True:
            # cache training data
            if not hasattr(self, "cached_X_"):
                self.cached_X_ = X
            if not hasattr(self, "cached_y_"):
                self.cached_y_ = y

            # check if input data has changed and force reset accordingly
            if not np.array_equal(self.cached_X_, X) or not np.array_equal(
                self.cached_y_, y
            ):
                self._initialize_problem(X, y)
            else:
                self._set_param_values()  # set parameter values

        solver_options = self.solver_options if self.solver_options is not None else {}
        if not isinstance(solver_options, dict):
            raise TypeError("solver_options must be a dictionary")

        self.coef_ = self._solve(X, y, solver_options, *args, **kwargs)
        self._set_intercept(X_offset, y_offset, X_scale)

        # return self for chaining fit and predict calls
        return self

    def _validate_params(self, X: ArrayLike, y: ArrayLike) -> None:
        """Validate hyper-parameters.

        Implement this in an estimator to check for valid hyper-parameters.
        """
        return

    def _set_param_values(self) -> None:
        """Set the values of cvxpy parameters from param attributes for warm starts."""
        return

    def _generate_params(self, X: ArrayLike, y: ArrayLike) -> Optional[SimpleNamespace]:
        """Return the named tuple of cvxpy parameters for optimization problem.

        The cvxpy Parameters must be given values when generating.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector

        Returns:
            NamedTuple of cvxpy parameters
        """
        return None

    def _generate_auxiliaries(
        self, X: ArrayLike, y: ArrayLike, beta: cp.Variable, parameters: SimpleNamespace
    ) -> Optional[SimpleNamespace]:
        """Generate any auxiliary variables or expressions necessary in defining the objective.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector
            beta (cp.Variable):
                cp.Variable representing the estimated coefs_
            parameters (SimpleNamespace):
                SimpleNamespace of cvxpy parameters.

        Returns:
            SimpleNamespace of cp.Variable for auxiliary variables
        """
        return None

    @abstractmethod
    def _generate_objective(
        self,
        X: ArrayLike,
        y: ArrayLike,
        beta: cp.Variable,
        auxiliaries: Optional[SimpleNamespace] = None,
        parameters: Optional[SimpleNamespace] = None,
    ) -> cp.Expression:
        """Define the cvxpy objective function represeting regression model.

        The objective must be stated for a minimization problem.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector
            auxiliaries (SimpleNamespace): optional
                SimpleNamespace with auxiliary cvxpy objects
            parameters (SimpleNamespace): optional
                SimpleNamespace with cp.Parameter objects

        Returns:
            cvpx Expression
        """
        return

    def _generate_constraints(
        self,
        X: ArrayLike,
        y: ArrayLike,
        auxiliaries: Optional[SimpleNamespace] = None,
        parameters: Optional[SimpleNamespace] = None,
    ) -> list[cp.constraints]:
        """Generate constraints for optimization problem.

        Args:
            X (ArrayLike):
                Covariate/Feature matrix
            y (ArrayLike):
                Target vector
            auxiliaries (SimpleNamespace): optional
                SimpleNamespace with auxiliary cvxpy objects
            parameters (SimpleNamespace): optional
                SimpleNamespace with cp.Parameter objects

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
        beta = cp.Variable(X.shape[1])
        parameters = self._generate_params(X, y)
        auxiliaries = self._generate_auxiliaries(X, y, beta, parameters)
        objective = self._generate_objective(X, y, beta, auxiliaries, parameters)
        constraints = self._generate_constraints(X, y, auxiliaries, parameters)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        self.canonicals_ = CVXCanonicals(
            problem=problem,
            objective=objective,
            beta=beta,
            auxiliaries=auxiliaries,
            parameters=parameters,
            constraints=constraints,
        )

    def _solve(self, X: ArrayLike, y: ArrayLike, solver_options: dict, *args, **kwargs):
        """Solve the cvxpy problem."""
        self.canonicals_.problem.solve(
            solver=self.solver, warm_start=self.warm_start, **solver_options
        )
        return self.canonicals_.beta.value


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

        objective = super()._generate_objective(X, y) + c0 * self._eta * cp.sum_squares(
            tikhonov_w @ self.beta_
        )

        return objective
