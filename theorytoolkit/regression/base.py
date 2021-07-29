"""Base classes for in-house linear regression estimators.

__author__ = "Luis Barroso-Luque, Fengyu Xie"
The classes make use of and follow the scikit-learn API.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from abc import ABCMeta, abstractmethod
import numpy as np
import cvxpy as cp
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model._base import  _rescale_data, _check_sample_weight


class Estimator(LinearModel, RegressorMixin, metaclass=ABCMeta):
    """
    Simple abstract estimator class based on sklearn linear model api to use
    different 'in-house'  solvers to fit a linear model. This should be used to
    create specific estimator classes by inheriting. New classes simply need to
    implement the _solve method to solve for the regression model coefficients.

    Keyword arguments are the same as those found in sklearn linear models.
    """

    def __init__(self, fit_intercept=False, normalize=False, copy_X=True):
        """
        fit_intercept : bool, default=True

        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
        Args:
            fit_intercept (bool):
                Whether the intercept should be estimated or not. If ``False``,
                the data is assumed to be already centered. normalize : bool
                default=False.
            normalize (bool):
                This parameter is ignored when ``fit_intercept`` is set to
                False.
                If True, the regressors X will be normalized before regression
                by subtracting the mean and dividing by the l2-norm.
            copy_X (bool):
                If ``True``, X will be copied; else, it may be overwritten.
        """
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

        self.coef_ = self._solve(X, y, *args, **kwargs)
        self._set_intercept(X_offset, y_offset, X_scale)

        # return self for chaining fit and predict calls
        return self

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
        X = feature_matrix
        y = target_vector

        if sample_weight is None:
            weights = np.ones(len(X))
        else:
            weights = np.array(sample_weight)

        # generate random partitions
        partitions = np.tile(np.arange(k), len(y) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(y)]

        all_cv = []
        #Compute 3 times and take average
        for n in range(5):
            ssr = 0

            for i in range(k):
                ins = (partitions != i)  # in the sample for this iteration
                oos = (partitions == i)  # out of the sample for this iteration

                self.fit(X[ins], y[ins],*args,\
                         sample_weight=weights[ins],\
                         **kwargs)
                res = (self.predict(X[oos]) - y[oos]) ** 2

                ssr += np.sum(res * weights[oos]) / np.average(weights[oos])

            cv = 1 - ssr / np.sum((y - np.average(y)) ** 2 * weights)

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
            optimal mu as a 1D np.array, and optimal cv score
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

        log_widths = np.array([ub-lb for ub,lb in log_mu_ranges],dtype=np.float64)
        log_centers = np.array([(ub+lb)/2 for ub,lb in log_mu_ranges],dtype=np.float64)
        #cvs_opt = 0

        for it in range(n_iter):
            for d in range(dim_mu):

                lb = log_centers[-d]-log_widths[-d]/2
                ub = log_centers[-d]+log_widths[-d]/2
                s = log_mu_steps[-d]

                #print("Current log centers:",log_centers)

                cur_mus = np.power(10,[log_centers for i in range(s)])
                cur_mus[:,-d] = np.power(10,np.linspace(lb,ub,s,dtype=np.float64))
                cur_cvs = [self.calc_cv_score(feature_matrix,target_vector,*args,\
                                              sample_weight=sample_weight,\
                                              mu=mu,**kwargs) for mu in cur_mus]
                i_max = np.nanargmax(cur_cvs)
                #cvs_opt = cur_cvs[i_max]
                #Update search conditions
                log_centers[-d] = np.linspace(lb,ub,s)[i_max]
                #For each iteration, shrink window by 4
                log_widths[-d] = log_widths[-d]/4

        return np.power(10,log_centers)

    @abstractmethod
    def _solve(self, X, y, *args, **kwargs):
        """Solve for the model coefficients."""
        return


class CVXEstimator(Estimator, metaclass=ABCMeta):
    """
    Base class for estimators using cvxpy with a sklearn interface.

    Note cvxpy can use one of many 3rd party solvers, default is most often
    CVXOPT. The solver can be specified by providing arguments to the cvxpy
    problem.solve function. And can be set by passing those arguments to the
    constructur of this class
    See documentation for more:
    https://ajfriendcvxpy.readthedocs.io/en/latest/tutorial/advanced/index.html#solve-method-options
    """

    def __init__(self, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """
        Args:
            fit_intercept (bool):
                Whether the intercept should be estimated or not.
                If False, the data is assumed to be already centered.
            normalize (bool):
                This parameter is ignored when fit_intercept is set to False.
                If True, the regressors X will be normalized before regression
                by subtracting the mean and dividing by the l2-norm.
                If you wish to standardize, please use StandardScaler before
                calling fit on an estimator with normalize=False
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
            **kwargs:
                Kewyard arguments passed to cvxpy solve.
                See docs linked above for more information.
        """
        self.warm_start = warm_start
        self.solver = solver
        self.solver_opts = kwargs
        self._problem, self._beta, self._X, self._y = None, None, None, None
        super().__init__(fit_intercept, normalize, copy_X)

    @abstractmethod
    def _initialize_problem(self, X, y):
        """Initialize cvxpy problem represeting regression model.

        Here only the coeficient variable Beta and X, y caching is done.
        """
        self._beta = cp.Variable(X.shape[1])
        self._X = X
        self._y = y

    def _get_problem(self, X, y):
        """Define and create cvxpy optimization problem"""
        if self._problem is None:
            self._initialize_problem(X, y)
        elif not np.array_equal(X, self._X) or not np.array_equal(y, self._y):
            self._initialize_problem(X, y)
        return self._problem

    def _solve(self, X, y, *args, **kwargs):
        """Solve the cvxpy problem."""
        problem = self._get_problem(X, y)
        problem.solve(solver=self.solver, warm_start=self.warm_start,
                      **self.solver_opts)
        return self._beta.value
