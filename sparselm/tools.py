"""A variety of tools for fitting linear regression models to polish CE fits."""

__author__ = "Luis Barroso-Luque"

import warnings
from functools import wraps
import numpy as np


def constrain_coefficients(indices, high, low=0.0):
    """Constrain a fit method to keep coefficients within a specified range.

    Decorator to enforce that a fit method fitting a cluster expansion that
    contains an EwaldTerm to constrain the dielectric constant to be positive
    and below the supplied value.

    Use this as a standard decorator with parameters:
    - At runtime: coefs = constrain_dielectric(indices, high, low)(fit_method)(X, y)
    - In fit_method definitions:
      @constrain_dielectric(indices, high, low)
      def your_fit_method(X, y):

    Args:
        indices (array or list):
            indices of coefficients to constrain
        high (float or array):
            upper bound for indices,
        low (float or array):
            lower bounds for indices
    """

    indices = np.array(indices)
    high = high * np.ones(len(indices)) if isinstance(high, float) else np.array(high)
    low = low * np.ones(len(indices)) if isinstance(low, float) else np.array(low)

    def decorate_fit_method(fit_method):
        """Decorate a fit method to constrain "dielectric constant".

        Args:
            fit_method (callable):
                the fit_method you will use to fit your cluster expansion.
                Must take the feature matrix X and target vector y as first
                arguments. (i.e. fit_method(X, y, *args, **kwargs)
        """
        @wraps(fit_method)
        def wrapped(X, y, *args, **kwargs):
            coefs = fit_method(X, y, *args, **kwargs)
            above_range = coefs[indices] > high
            below_range = coefs[indices] < low

            # TODO do not set features to zero, do the fit without them instead
            if sum(above_range) > 0 or sum(below_range) > 0:
                X_, y_ = X.copy(), y.copy()
                y_ -= np.sum(X_[:, indices[above_range]] * high[above_range], axis=1)
                X_[:, indices[above_range]] = 0.0
                y_ -= np.sum(X_[:, indices[below_range]] * low[below_range], axis=1)
                X_[:, indices[below_range]] = 0.0
                coefs = fit_method(X_, y_, *args, **kwargs)
                coefs[indices[above_range]] = high[above_range]
                coefs[indices[below_range]] = low[below_range]

            # check if new coeficients are now out of range
            above_range = coefs[indices] > high
            below_range = coefs[indices] < low
            if sum(above_range) > 0 or sum(below_range) > 0:
                warnings.warn(
                    "Running the constrained fit has resulted in new out of range "
                    "coefficients that were not so in the unconstrained fit.\n"
                    "Double check the sensibility of the bounds you provided!",
                    RuntimeWarning
                )

            return coefs
        return wrapped
    return decorate_fit_method
