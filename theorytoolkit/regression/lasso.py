"""L1 regularization least squares solver."""

__author__ = "William Davidson Richard, Fengyu Xie"

import numpy as np
import warnings
import math
#Switch to cvxpy optimization.
import cvxpy as cp
from .base import BaseEstimator


class LassoEstimator(BaseEstimator):
    """
    Lasso Estimator implemented with cvxpy.
    """

    def __init__(self):
        super().__init__()

    def fit(self, feature_matrix, target_vector, sample_weight=None,\
            mu=None, log_mu_ranges=[(-1,6)], log_mu_steps = [8]):
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
            Optimized mu for storage convenience.
            Fitter coefficients storeed in self.coef_.
        """
        if isinstance(mu,(int,float)):
            mu = [float(mu)]
        #Always call super().fit because this contains preprocessing of matrix 
        #and vector, such as centering and weighting!           
        if mu is None or len(mu)!=1:
            mu = super().optimize_mu(feature_matrix, target_vector,
                                     sample_weight=sample_weight,
                                     dim_mu=1,
                                     log_mu_ranges=log_mu_ranges,
                                     log_mu_steps=log_mu_steps)
            if mu[0]<=np.power(10,log_mu_ranges[0][0]):
                warnings.warn("Minimun allowed mu taken!")
            if mu[0]>=np.power(10,log_mu_ranges[0][1]):
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
        if mu[0]<0:
            raise ValueError("Mu can not be negative!")
         
        A = feature_matrix.copy()
        b = target_vector.copy()
        n = A.shape[0]
        d = A.shape[1]
        
        w = cp.Variable((d,))
        z1 = cp.Variable((d,),pos=True)
        constraints = [z1>=w, z1>=-w]
        #Hierarchy constraints are not supported by regularization without L0

        # Cost function
        L = cp.sum_squares(A@w-b)+mu[0]*cp.sum(z1)

        prob = cp.Problem(cp.Minimize(L), constraints)
        prob.solve()

        return w.value
