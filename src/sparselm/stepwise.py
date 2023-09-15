"""Stepwise model selection for piece-wise fitting."""

from __future__ import annotations

__author__ = "Fengyu Xie"

from itertools import chain

import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel, _check_sample_weight
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted


# BaseComposition makes sure that StepwiseEstimator can be correctly cloned.
def _indices_no_overlap_and_continuous(indices):
    scope = sorted(set(chain(*indices)))
    return sorted(chain(*indices)) == scope and scope == list(range(len(scope)))


def _first_step_fit_intercept_only(steps):
    for sid, (_, estimator) in enumerate(steps):
        if hasattr(estimator, "estimator"):
            # Is a searcher such as GridSearchCV.
            fit_intercept = estimator.estimator.fit_intercept
        else:
            fit_intercept = estimator.fit_intercept
        if fit_intercept and sid > 0:
            return False
    return True


def _no_nested_stepwise(steps):
    for _, estimator in steps:
        if isinstance(estimator, StepwiseEstimator):
            return False
    return True


class StepwiseEstimator(_BaseComposition, RegressorMixin, LinearModel):
    """A composite estimator used to do stepwise fitting.

    The first estimator in the composite will be used to fit
    certain features (a piece of the feature matrix) to the
    target vector, and the residuals are fitted to the rest
    of features by using the next estimators in the composite.

    Each estimator can be either a CVXEstimator, a GridSearchCV or
    a LineSearchCV.

    Args:
        steps (list[(str, CVXEstimator)]):
            A list of step names and the CVXEstimators to use
            for each step. StepwiseEstimator cannot be used as
            a member of StepwiseEstimator.
            An estimator will fit the residuals of the previous
            estimator fits in the list.
        estimator_feature_indices (tuple[tuple[int]]):
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

    Notes:
        1. Do not use GridSearchCV or LineSearchCV to search a StepwiseEstimator!

        2. No nesting is allowed for StepwiseEstimator, which means no step of a
        StepwiseEstimator can be a StepwiseEstimator.

        3. Since stepwise estimator requires specifying a list of feature indices for
        each step estimator, it requires fixing n_features_in_ before fitting, which
        violates sklearn convention for a regressor. Therefore, StepwiseEstimator is
        not checked by sklearn check_estimator method, and there is no guarantee that it
        is fully compatible with all scikit-learn features.
    """

    def __init__(
        self,
        steps,
        estimator_feature_indices,
    ):
        self.steps = steps
        # The estimator_feature_indices saved must be tuple because in
        # sklearn.base.clone, a cloned object is checked by pointer, rather than
        # by value.
        self.estimator_feature_indices = estimator_feature_indices

    # These parameters settings does not need to be called externally.
    def get_params(self, deep=True):
        """Get parameters of all estimators in the composite.

        Args:
            deep(bool):
                If True, will return the parameters for estimators in
                composite, and their contained sub-objects if they are
                also estimators.
        """
        return self._get_params("steps", deep=deep)

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
        self._set_params("steps", **params)
        return self

    @staticmethod
    def _get_estimator_coef(estimator):
        check_is_fitted(estimator)
        if hasattr(estimator, "best_estimator_"):
            return estimator.best_estimator_.coef_.copy()
        elif hasattr(estimator, "coef_"):
            return estimator.coef_.copy()
        else:
            raise ValueError(f"Estimator {estimator} is not a valid linear model!")

    @staticmethod
    def _get_estimator_intercept(estimator):
        check_is_fitted(estimator)
        if hasattr(estimator, "best_estimator_"):
            return estimator.best_estimator_.intercept_
        elif hasattr(estimator, "intercept_"):
            return estimator.intercept_
        else:
            raise ValueError(f"Estimator {estimator} is not a valid linear model!")

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        sample_weight: NDArray | None = None,
        *args,
        **kwargs,
    ):
        """Prepare fit input with sklearn help then call fit method.

        Args:
            X (NDArray):
                Training data of shape (n_samples, n_features).
            y (NDArray):
                Target values. Will be cast to X's dtype if necessary
                of shape (n_samples,) or (n_samples, n_targets)
            sample_weight (NDArray):
                Individual weights for each sample of shape (n_samples,)
                default=None
            *args:
                Positional arguments passed to _fit method
            **kwargs:
                Keyword arguments passed to _fit method
        Returns:
            instance of self
        """
        # Check estimators and feature indices.
        if not _indices_no_overlap_and_continuous(self.estimator_feature_indices):
            raise InvalidParameterError(
                f"Given feature indices:"
                f" {self.estimator_feature_indices}"
                f" are not continuous and non-overlapping"
                f" series starting from 0!"
            )
        if not _first_step_fit_intercept_only(self.steps):
            raise InvalidParameterError(
                "Only the first estimator in steps is allowed" " to fit intercept!"
            )
        if not _no_nested_stepwise(self.steps):
            raise InvalidParameterError(
                "StepwiseEstimator should not be nested with"
                " another StepwiseEstimator!"
            )

        self.n_features_in_ = len(list(chain(*self.estimator_feature_indices)))

        # Set ensute_2d to True and reset to False so that it triggers number of
        # features checking.
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            y_numeric=True,
            multi_output=True,
            reset=False,
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        residuals = y.copy()

        self.coef_ = np.empty(X.shape[1])
        self.coef_.fill(np.nan)
        for (_, estimator), scope in zip(self.steps, self.estimator_feature_indices):
            # np.array indices should not be tuple.
            estimator.fit(
                X[:, list(scope)],
                residuals,
                *args,
                sample_weight=sample_weight,
                **kwargs,
            )
            self.coef_[list(scope)] = self._get_estimator_coef(estimator)
            residuals = residuals - estimator.predict(X[:, list(scope)])
            # Only the first estimator is allowed to fit intercept.
        if hasattr(self.steps[0][1], "estimator"):
            fit_intercept = self.steps[0][1].estimator.fit_intercept
        else:
            fit_intercept = self.steps[0][1].fit_intercept
        if fit_intercept:
            self.intercept_ = self._get_estimator_intercept(self.steps[0][1])
        else:
            self.intercept_ = 0.0

        # return self for chaining fit and predict calls
        return self
