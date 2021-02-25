""" L1L0 regularization least square solver. """

__author__ = "Fengyu Xie"

import numpy as np
import warnings
import math
import cvxpy as cp
from .base import BaseEstimator
from .utils import time_limit, TimeoutException

class L1L0Estimator(BaseEstimator):
    """
    L1L0 estimator. Removed Gurobi dependency!
    Uses any solver it has in hand. If nothing, will use ECOS_BB.
    ECOS_BB is only supported after cvxpy-1.1.6. It will not be given 
    automatically pulled up by cvxpy, and has to be explicitly called
    as a parameter of problem.solve(solver='ECOS_BB').
    Installation of Gurobi is no longer a must, but highly recommended,
    since ECOS_BB can be very slow. You may also choose CPLEX, etc.
    """
    def __init__(self):
        super().__init__()

    def fit(self,feature_matrix, target_vector, sample_weight=None, hierarchy=None,\
            mu=None, log_mu_ranges=[(-8,0),(-6,1)], log_mu_steps = [9,8],\
            M=100, tlimit=300):
        """
        Inputs:
            feature_matrix(2D array like):
                Featire matrix of structure pool
            target_vector(2D array like):
                Fitted physical quantities (usually normalized)
                Must be of same length to feature matrix.
            hierarchy(2D array like):
                A list of integers storing hirarchy relation between clusters.
                Each sublist contains indices of higher order correlation functions 
                that contains this correlation function.
                If none given, will not add hierarchy constraints.
            sample_weight(1D array-like):
                Weight of each entry in the feature matrix.
                Must be of the same length as feature matrix.
            mu(1D array-like of length 1 or None):
                A tuple of (mu0,mu1). L1L0 loss function is:
                L = ||Xw-y||^2 + mu0 * ||w||_0 + mu1 * ||w||_1 
            log_mu_ranges(None|List[(float,float)]):
                allowed optimization ranges of log(mu). If not provided, will be guessed.
                But I still highly recommend you to give this based on your experience.
            log_mu_steps(None|List[int]):
                Number of steps to search in each log_mu coordinate. Optional, but also
                recommeneded.
            M(float):
                maximum allowed absolute value of any coefficient. Used 
                as a control parameter in L0 term.
            tlimit(float):
                Time limit of each L1L0 solve in seconds. If time limit is exceeded, 
                will raise an error.
        Return:
            Optimized mu.      
            Fitter coefficients are also stored in self.coef_.       
        """ 
        #Always call super().fit because this contains preprocessing of matrix 
        #and vector, such as centering and weighting!
        if mu is None or len(mu)!=2:
            mu = super().optimize_mu(feature_matrix, target_vector,
                                     sample_weight=sample_weight,
                                     hierarchy=hierarchy,
                                     dim_mu=2, 
                                     log_mu_ranges=log_mu_ranges,
                                     log_mu_steps=log_mu_steps,
                                     M=M, tlimit=tlimit)
            if mu[0]<=np.power(10,float(log_mu_ranges[0][0])):
                warnings.warn("Minimun allowed mu_0 taken!")
            if mu[0]>=np.power(10,float(log_mu_ranges[0][1])):
                warnings.warn("Maximum allowed mu_0 taken!")
            if mu[1]<=np.power(10,float(log_mu_ranges[1][0])):
                warnings.warn("Minimun allowed mu_1 taken!")
            if mu[1]>=np.power(10,float(log_mu_ranges[1][1])):
                warnings.warn("Maximum allowed mu_1 taken!")

        super().fit(feature_matrix, target_vector,
                    sample_weight=sample_weight, 
                    hierarchy=hierarchy,
                    mu=mu, M=M, tlimit=tlimit)
        return mu

    def _solve(self,feature_matrix,target_vector,hierarchy=None,mu=[0,0],\
               M=100,tlimit=300):
        """
        Called by super().fit. Solves an L1L0 model with hierarchy constraints.
        Inputs:
            M(float):
                L0 slack number. Should be larger than any coefficient's absolute value.
            tlimit(float):
                Optimization cutoff time. By default set to 300s.
        Outputs:
            fitted coefficients, in np.array.
        """
        if mu[0]<0 or mu[1]<0:
            raise ValueError("Mu can not be negative!")

        A = feature_matrix.copy()
        b = target_vector.copy()
        n = A.shape[0]
        d = A.shape[1]
        
        w = cp.Variable((d,))
        z0 = cp.Variable((d,),integer=True)
        z1 = cp.Variable((d,),pos=True)
        constraints = [0<=z0, z0<=1, M*z0>=w, M*z0>=-w, z1>=w, z1>=-w]
        #Hierarchy constraints.
        if hierarchy is not None:
            for sub_id,high_ids in enumerate(hierarchy):
                for high_id in high_ids:
                    constraints.append(z0[high_id]<=z0[sub_id])
        # Cost function
        L = cp.sum_squares(A@w-b)+mu[0]*cp.sum(z0)+mu[1]*cp.sum(z1)

        prob = cp.Problem(cp.Minimize(L), constraints)
        #Use any solver it has in hand. If nothing, will use ECOS_BB
        #Installation of Gurobi is no longer a must, but recommended
        # You may also choose CPLEX, etc.
        try:
            with time_limit(tlimit):
                prob.solve()
        except:
            warnings.warn("No compatible MIQP solver found. Trying with ECOS_BB.")
            try:
                with time_limit(tlimit):
                    prob.solve(solver='ECOS_BB')
            except:
                raise TimeoutException("All solvers timed out! You may consider larger tlimit.")

        return w.value
