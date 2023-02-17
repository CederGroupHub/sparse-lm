"""Composite estimator for piece-wise fitting.
"""

__author__ = "Fengyu Xie"

import re
from itertools import chain

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel


class CompositeEstimator(RegressorMixin, LinearModel):
    """A composite estimator used to do piece-wise fitting.

    The first estimator in the composite will be used to fit
    certain features (a piece of the feature matrix, called scope)
    to the target vector, and the residuals are fitted to the rest
    of features by using the next estimators in the composite.
    """

    def __init__(self, estimators, estimator_scopes):
        """Initialize estimator.

        Args:
            estimators(list[CVXEstimator]):
                A list of CVXEstimators to add. CompositeEstimator
                cannot be a member of CompositeEstimator.
                An estimator will fit the residuals leftover by
                the previous estimator fits in the list.
            estimator_scopes(list[list[int]]):
                Scope of each estimator, which means the indices of
                features in the scope (features[:, scope]) will be
                fitted to the residual using the corresponding estimator.
                Notice:
                   If estimators in the composite requires hierarchy
                   or groups, the indices in the groups or hierarchy
                   must be adjusted such that they correspond to the groups
                   or hierarchy relations in the part of features sliced
                   by scope.
                   For example, consider original groups = [0, 1, 1, 2, 2],
                   and an estimator has scope = [3, 4], then the estimator
                   should be initialized with group = [0, 0].
                   You are fully responsible to initialize the estimators
                   with correct hierarchy, groups and other parameters before
                   wrapping them up with the composite!
        """
        if len(estimators) != len(estimator_scopes):
            raise ValueError("Must specify a scope for each estimator!")
        full_scope = sorted(set(chain(*estimator_scopes)))
        if full_scope != sorted(chain(*estimator_scopes)) or full_scope != list(
            range(len(full_scope))
        ):
            raise ValueError(
                f"Estimator scopes: {estimator_scopes}"
                f" overlapped or not continuous!"
            )
        for estimator in estimators:
            if isinstance(estimator, CompositeEstimator):
                raise ValueError(
                    "Cannot add a CompositeEstimator" " into a CompositeEstimator!"
                )

        self._estimators = estimators
        self._estimator_scopes = [
            np.array(scope, dtype=int).tolist() for scope in estimator_scopes
        ]
        self._full_scope = full_scope
        # Only the first estimator is allowed to fit intercept.
        if len(self._estimators) > 1:
            for estimator in self._estimators[1:]:
                estimator.fit_intercept = False

    # Overwrite the class method in BaseEstimator as an object method.
    def _get_param_names(self):
        """Get parameter names for all estimators in the composite."""
        all_params = [estimator._get_param_names() for estimator in self._estimators]
        all_names = []
        for est_id, names in enumerate(all_params):
            for name in names:
                all_names.append(name + f"_{est_id}")
        return all_names

    def get_params(self, deep=True):
        """Get parameters of all estimators in the composite.

        Args:
            deep(bool):
                If True, will return the parameters for estimators in
                composite, and their contained sub-objects if they are
                also estimators.
        """
        out = dict()
        est_params = [estimator.get_params(deep=deep) for estimator in self._estimators]
        for name in self._get_param_names():
            real_name, est_ind = re.match(r"(.*)_(\d+)$", name).groups()
            out[name] = est_params[int(est_ind)].get(real_name)
        return out

    def set_params(self, **params):
        """Set parameters for each estimator in the composite.

        This will be called when model selection optimizes
        all hyper parameters.
        Args:
            params: A Dictionary of parameters. Each parameter
            name must end with an underscore and a number to specify
            on which estimator in the composite the parameter is
            going to be set.
            Remember only to set params you wish to optimize!
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        params_for_estimators = [{} for _ in range(len(self._estimators))]
        for name, value in params.items():
            real_name, est_ind = re.match(r"(.*)_(\d+)$", name).groups()
            params_for_estimators[int(est_ind)][real_name] = value
        for estimator_params, estimator in zip(params_for_estimators, self._estimators):
            estimator.set_params(**estimator_params)

        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        *args,
        **kwargs,
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
        # Check scope overlap.
        X = np.asarray(X)
        y = np.asarray(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        residuals = y.copy()

        if self._full_scope != list(range(X.shape[1])):
            raise ValueError(
                f"Full scope: {self._full_scope} does not cover"
                f" all {X.shape[1]} features!"
            )

        self.coef_ = np.empty(X.shape[1])
        self.coef_.fill(np.nan)
        for estimator, scope in zip(self._estimators, self._estimator_scopes):
            estimator.fit(X[:, scope], residuals, sample_weight, *args, **kwargs)
            self.coef_[scope] = estimator.coef_.copy()
            residuals = residuals - estimator.predict(X[:, scope])
            # Only the first estimator is allowed to fit intercept.
        if self._estimators[0].fit_intercept:
            self.intercept_ = self._estimators[0].intercept_
        else:
            self.intercept_ = 0.0

        # return self for chaining fit and predict calls
        return self
