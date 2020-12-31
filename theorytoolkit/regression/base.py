
__author__ = "Luis Barroso-Luque, Fengyu Xie"

from abc import ABC, abstractmethod
import numpy as np
from smol.exceptions import NotFittedError

class BaseEstimator(ABC):
    """
    A simple estimator class to use different 'in-house'  solvers to fit a
    cluster-expansion. This should be used to create specific estimator classes
    by inheriting. New classes simple need to implement the solve method.
    The methods have the same signatures as most scikit-learn regressors, such
    that those can be directly used instead of this to fit a cluster-expansion
    The base estimator does not fit. It only has a predict function for
    Expansions where the user supplies the ecis.
    """

    def __init__(self):
        self.coef_ = None

    def fit(self, feature_matrix, target_vector, *args, sample_weight=None,
            **kwargs):
        """
        Prepare fit input then fit. First, weighting. Then centering.
        Point terms are not separated for linear regression. (Not implemented
        yet).
        """
        if sample_weight is not None:
            feature_matrix = feature_matrix * sample_weight[:, None] ** 0.5
            target_vector = target_vector * sample_weight ** 0.5

        feature_av = np.mean(feature_matrix,axis=0)
        feature_centered = feature_matrix - feature_av
        traget_av = np.mean(target_vector)
        target_centered = target_vector - target_av

        #TODO: implement separate fitting of single-point terms. Is it necessary?

        coef_ = self._solve(feature_centered, target_centered,
                                 *args, **kwargs)
        self.coef_ = coef_.copy()
        self.coef_[0] += (target_av-np.dot(feature_av,coef_))

    def predict(self, feature_matrix):
        """Predict a new value based on fit"""
        if self.coef_ is None:
            raise NotFittedError('This estimator has not been fitted.')
        return np.dot(feature_matrix, self.coef_)

    def calc_cv_score(self, feature_matrix, target_vector, *args, sample_weight=None,\
                      k=5, **kwargs):
        """
        Args:
            feature_matrix: sensing matrix (scaled appropriately)
            target_vector: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        if sample_weight is None:
            weights = np.ones(len(X[:, 0]))
        else:
            weights = np.array(sample_weight)

        # generate random partitions
        partitions = np.tile(np.arange(k), len(y) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(y)]

        all_cv = []
        #Compute 3 times and take average
        for n in range(3):
            ssr = 0
            ssr_uw = 0
            for i in range(k):
                ins = (partitions != i)  # in the sample for this iteration
                oos = (partitions == i)  # out of the sample for this iteration
    
                self.fit(feature_matrix[ins], target_vector[ins],*args,\
                         sample_weight=weights[ins],\
                         **kwargs)
                res = (self.predict(feature_matrix[oos]) - target_vector[oos]) ** 2
                ssr += np.sum(res * weights[oos]) / np.average(weights[oos])
                ssr_uw += np.sum(res)
            cv = 1 - ssr / np.sum((y - np.average(y)) ** 2)
            all_cv.append(cv)
        return np.average(all_cv)

    def optimize_mu(self,feature_matrix, target_vector,*args,sample_weight=None,\
                    dim_mu=0,n_iter=2,\
                    log_mu_ranges=None,log_mu_steps=None,\
                    **kwargs):
        """
        If the estimator supports mu parameters, this method provides a quick, coordinate
        descent method to find the optimal mu for the model, by minimizing cv (maximizing
        cv score).
        Any mu should be defined as a 1 dimensional array of length dim_mu, and the 
        optimized log_mu's are constrained within log_mu_ranges.
        The optimization will always start from the last dimension of mu, so in L0L1 or 
        L0L2, make sure that the last mu is your mu_1 or mu_2.

        Inputs:
            dim_mu(int):
                length of arrayLike mu.
            n_iter(int):
                number of coordinate descent iterations to do. By default, will do 3
                iterations.
            log_mu_ranges(None|List[(float,float)]):
                allowed optimization ranges of log(mu). If not provided, will be guessed.
                But I still highly recommend you to give this based on your experience.
            log_mu_steps(None|List[int]):
                Number of steps to search in each log_mu coordinate. If not given, 
                Will set to 11 for each log_mu coordinate.
        Outputs:
            optimal mu as a 1D np.array.
        """
        if dim_mu==0:
            #No optimization needed.
            return None
        if log_mu_ranges is not None and len(log_mu_ranges)!=dim_mu:
            raise ValueError('Length of log(mu) search ranges does not match number of mus!')
        if log_mu_steps is not None and len(log_mu_steps)!=dim_mu:
            raise ValueError('Length of log(mu) search steps does not match number of mus!')
        if log_mu_ranges is None:
            log_mu_ranges = [(-5,5) for i in range(dim_mu)]
        if log_mu_steps is None:
            log_mu_steps = [11 for i in range(dim_mu)]

        log_widths = np.array([ub-lb for ub,lb in log_mu_ranges])
        log_centers = np.array([(ub+lb)/2 for ub,lb in log_mu_ranges])

        for it in range(n_iter):
            for d in range(dim_mu):
                lb = log_centers[-d]-log_widths[-d]/2
                ub = log_centers[-d]+log_widths[-d]/2
                s = log_mu_steps[-d]
                cur_mus = np.power(10,[log_centers for i in range(s)])
                cur_mus[:,-d] = np.power(10,np.linspace(lb,ub,s))
                cur_cvs = [self.calc_cv_score(feature_matrix,target_vector,*args,\
                                              sample_weight=sample_weight,\
                                              mu=mu,**kwargs) for mu in cur_mus]
                i_max = np.nanargmax(cur_cvs)
                #Update search conditions
                log_centers[-d] = np.linspace(lb,ub,s)[i_max]
                #For each iteration, shrink window by 4
                log_widths[-d] = log_widths[-d]/4

        return np.power(10,log_centers)

    @abstractmethod
    def _solve(self, feature_matrix, target_vector, *args, **kwargs):
        """Solve for the learn coefficients."""
        return
