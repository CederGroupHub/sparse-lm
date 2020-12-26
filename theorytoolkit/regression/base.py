
__author__ = "Luis Barroso-Luque"

from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model.base import  _rescale_data, _check_sample_weight


class Estimator(LinearModel, RegressorMixin, metaclass=ABCMeta):
    """
    Simple abstract estimator class based on sklearn linear model api to use
    different 'in-house'  solvers to fit a linear model. This should be used to
    create specific estimator classes by inheriting. New classes simple need to
    implement the solve method.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.coef_, self.intercept_ = None, None

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        """Prepare fit input with sklearn help then call fit method.

        Args:
            X (array-like):
                Training data of shape (n_samples, n_features).
            y (array-like):
                Target values. Will be cast to X's dtype if necessary
                of shape (n_samples,) or (n_samples, n_targets)
            sample_weight (array-like):
                Individual weights for each sample of shape (n_samples,)
                default=None
            *args:
                Positional arguments passed to _fit method
            **kwargs:
                Keyword arguments passed to _fit method

        Returns:
            instance of self
        """
        X, y = self._validate_data(X, y, accept_sparse=False,
                                   y_numeric=True, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight,
            return_mean=True)

        if sample_weight is not None:
            X, y = _rescale_data(X, y, sample_weight)

        self._set_intercept(X_offset, y_offset, X_scale)
        self.coef_ = self._solve(X, y, *args, **kwargs)

        return self

    @abstractmethod
    def _solve(self, X, y, *args, **kwargs):
        """Solve for the learn coefficients."""
        return


class CVXEstimator(Estimator):
    """
    Wrapper base class for estimators using cvxpy with a sklearn interface.

    Note cvxpy can use one of many 3rd party solvers, default is most often
    CVXOPT. The solver can be specified by providing arguments to the cvxpy
    problem.solve function. And can be set by passing those arguments to the
    constructur of this class
    See documentation for more:
    https://ajfriendcvxpy.readthedocs.io/en/latest/tutorial/advanced/index.html#solve-method-options
    """
    
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 warm_start=False, solver=None, verbose=False, **kwargs):
        self.problem, self.beta = self._initialize_problem()
        self.warm_start = warm_start
        self._solver_opts = {'solver': solver, 'verbose': verbose, **kwargs}
        super().__init__(fit_intercept, normalize, copy_X)

    @abstractmethod
    def _initialize_problem(self, *args, **kwargs):
        """Define and create cvxpy optimization problem"""
        return ()

    def _solve(self):
        self.problem.solve(warm_start=self.warm_start, **self._solver_opts)
        return self.beta.value
