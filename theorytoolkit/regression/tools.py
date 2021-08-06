"""A variety of tools for fitting linear regression models to polish CE fits."""

__author__ = "Luis Barroso-Luque, Fengyu Xie"
__credits__ = "William Davidson Richard"

from functools import wraps
import numpy as np


def constrain_dielectric(max_dielectric, ewald_ind=-1):
    """Constrain a fit method to keep dieletric 0<= e < max_dielectric.

    Decorator to enforce that a fit method fitting a cluster expansion that
    contains an EwaldTerm to constrain the dielectric constant to be positive
    and below the supplied value.

    If the dielectric (inverse of the Ewald eci) is negative or above the max
    dielectric, the decorator will force the given fit_method to refit to the
    target vector with the Ewald interactions times the max dielectric
    subtracted out.

    Use this as a standard decorator with parameters:
    - At runtime: ecis = constrain_dielectric(max_dielectric)(fit_method)(X, y)
    - In fit_method definitions: @constrain_dielectric(max_dielectric)
                                 def your_fit_method(X, y):

    Args:
        max_dielectric (float):
            Value of maximum dielectric constant to constrain by.
        ewald_ind (int):
            Index of column of Ewald interaction features in the feature matrix
    """
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
            ecis = fit_method(X, y, *args, **kwargs)
            if ecis[ewald_ind] < 1.0 / max_dielectric:
                X_, y_ = X.copy(), y.copy()
                y_ -= X_[:, ewald_ind] / max_dielectric
                X_[:, ewald_ind] = 0
                ecis = fit_method(X_, y_, *args, **kwargs)
                ecis[ewald_ind] = 1.0 / max_dielectric
            return ecis
        return wrapped
    return decorate_fit_method


def calc_cv_score(estimator, X, y, k=5, sample_weight=None, **kwargs):
    """
    Partition the sample into k partitions, calculate out-of-sample
    variance for each of these partitions, and add them together

    Args:
        estimator: 
            Estimator class with fit and predict methods
            feature_matrix: feature matrix (scaled appropriately)
        k (int):
            number of partitions, CV fold
        X (ndarray):
            feature matrix for fit
        y (ndarray):
            data to fit (scaled appropriately)
        **kwargs:
            key word arguments to pass to estimator solve function.

    Returns:
        CV score
    """
    if sample_weight is None:
        weights = np.ones(len(X))
    else:
        weights = np.array(sample_weight)

    # generate random partitions
    partitions = np.tile(np.arange(k), len(y) // k + 1)
    np.random.shuffle(partitions)
    partitions = partitions[:len(y)]

    all_cv = []
    # Compute 5 times and take average, why 5?
    for n in range(5):
        ssr = 0
        for i in range(k):
            ins = (partitions != i)  # in the sample for this iteration
            oos = (partitions == i)  # out of the sample for this iteration

            estimator.fit(
                X[ins], y[ins], sample_weight=weights[ins], **kwargs)
            res = (estimator.predict(X[oos]) - y[oos]) ** 2
            ssr += np.sum(res * weights[oos]) / np.average(weights[oos])

        cv = 1 - ssr / np.sum((y - np.average(y)) ** 2 * weights)
        all_cv.append(cv)

    return np.average(all_cv)


# TODO not sure if this will work for 2 hyperparameters...
def optimize_alpha(estimator, feature_matrix, target_vector, sample_weight=None,
                   dim_alpha=0, n_iter=2, log_alpha_ranges=None,
                   log_alpha_steps=None, **kwargs):
    """
    If the estimator supports alpha parameters, this method provides a quick,
    coordinate descent method to find the optimal mu for the model, by
    minimizing cv (maximizing cv score).
    Any alpha should be defined as a 1 dimensional array of length dim_alpha,
    and the optimized log_alphas's are constrained within log_alpha_ranges.
    The optimization will always start from the last dimension of alpha, so in
    L0L1 or L0L2, make sure that the last alpha is your alpha_1 or alpha_2.

    Inputs:
        estimator:
            Estimator class with fit and predict methods
        X:
            feature matrix (scaled appropriately)
        y:
            data to fit (scaled appropriately)
        dim_alpha(int):
            length of arrayLike alpha.
        n_iter(int):
            number of coordinate descent iterations to do. By default, will do 3
            iterations.
        log_alpha_ranges(None|List[(float,float)]):
            allowed optimization ranges of log(mu). If not provided, will be guessed.
            But I still highly recommend you to give this based on your experience.
        log_alpha_steps(None|List[int]):
            Number of steps to search in each log_mu coordinate. If not given,
            Will set to 11 for each log_mu coordinate.
    Outputs:
        optimal mu as a 1D np.array, and optimal cv score
    """
    if dim_alpha==0:
        #No optimization needed.
        return None
    if log_alpha_ranges is not None and len(log_alpha_ranges)!=dim_alpha:
        raise ValueError(
            'Length of log(mu) search ranges does not match number of mus!')
    if log_alpha_steps is not None and len(log_alpha_steps)!=dim_alpha:
        raise ValueError(
            'Length of log(mu) search steps does not match number of mus!')
    if log_alpha_ranges is None:
        log_alpha_ranges = [(-5, 5) for i in range(dim_alpha)]
    if log_alpha_steps is None:
        log_alpha_steps = [11 for i in range(dim_alpha)]

    log_widths = np.array(
        [ub - lb for ub, lb in log_alpha_ranges], dtype=np.float64)
    log_centers = np.array(
        [(ub+lb) / 2 for ub, lb in log_alpha_ranges], dtype=np.float64)
    #cvs_opt = 0

    for it in range(n_iter):
        for d in range(dim_alpha):

            lb = log_centers[-d]-log_widths[-d]/2
            ub = log_centers[-d]+log_widths[-d]/2
            s = log_alpha_steps[-d]

            #print("Current log centers:",log_centers)

            cur_alphas = np.power(10, [log_centers for _ in range(s)])
            cur_alphas[:, -d] = np.power(
                10, np.linspace(lb, ub, s, dtype=np.float64))

            cur_cvs = []
            for alpha in cur_alphas:
                estimator.alpha = alpha
                cur_cvs.append(
                    calc_cv_score(estimator, feature_matrix, target_vector,
                                  sample_weight=sample_weight, **kwargs))

            i_max = np.nanargmax(cur_cvs)
            #cvs_opt = cur_cvs[i_max]
            #Update search conditions
            log_centers[-d] = np.linspace(lb, ub, s)[i_max]
            #For each iteration, shrink window by 4
            log_widths[-d] = log_widths[-d]/4

    return np.power(10, log_centers)
