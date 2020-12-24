"""L1 regularization least squares solver."""

__author__ = "William Davidson Richard"

import numpy as np
import warnings
import math
from cvxopt import matrix, spdiag, mul, div, sqrt
from cvxopt import blas, lapack, solvers
from .base import BaseEstimator


class WDRLasso(BaseEstimator):
    """
    Estimator implementing the written l1regs cvx based solver. Written
    by WD Richards. This is not tested, so use at your own risk.
    """

    def __init__(self):
        warnings.warn('This class will be deprecated soon, so do not get too '
                      'attached to it.\nConsider using 3rd party estimators '
                      'such as scikit learn.', category=DeprecationWarning,
                      stacklevel=2)
        super().__init__()
        self.mus = None
        self.cvs = None

    def fit(self, feature_matrix, target_vector, sample_weight=None, mu=None):
        """Fit the estimator."""
        if mu is None:
            mu = self._get_optimum_mu(feature_matrix, target_vector,
                                      sample_weight)
        super().fit(feature_matrix, target_vector,
                    sample_weight=sample_weight, mu=mu)

    def _solve(self, feature_matrix, target_vector, mu):
        """
        X and y should already have been adjusted to account for weighting.
        """

        # Maybe its cleaner to use importlib to try and import these?
        solvers.options['show_progress'] = False

        X1 = matrix(feature_matrix)
        b = matrix(target_vector * mu)
        return (np.array(l1regls(X1, b)) / mu).flatten()

    def _calc_cv_score(self, mu, X, y, weights, k=5):
        """
        Args:
            mu: weight of error in bregman
            X: sensing matrix (scaled appropriately)
            y: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        if weights is None:
            weights = np.ones(len(X[:, 0]))

        # generate random partitions
        partitions = np.tile(np.arange(k), len(y) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(y)]

        ssr = 0
        ssr_uw = 0
        for i in range(k):
            ins = (partitions != i)  # in the sample for this iteration
            oos = (partitions == i)  # out of the sample for this iteration

            self.fit(X[ins], y[ins], weights[ins], mu)
            res = (np.dot(X[oos], self.coef_) - y[oos]) ** 2
            ssr += np.sum(res * weights[oos]) / np.average(weights[oos])
            ssr_uw += np.sum(res)

        cv = 1 - ssr / np.sum((y - np.average(y)) ** 2)
        return cv

    def _get_optimum_mu(self, X, y, weights, k=5, min_mu=0.1, max_mu=6):
        """
        Finds the value of mu that maximizes the cv score
        """
        mus = list(np.logspace(min_mu, max_mu, 10))
        cvs = [self._calc_cv_score(mu, X, y, weights, k) for mu in mus]

        for _ in range(2):
            i = np.nanargmax(cvs)
            if i == len(mus) - 1:
                warnings.warn('Largest mu chosen. You should probably'
                              ' increase the basis set')
                break

            mu = (mus[i] * mus[i + 1]) ** 0.5
            mus[i + 1:i + 1] = [mu]
            cvs[i + 1:i + 1] = [self._calc_cv_score(mu, X, y, weights, k)]

            mu = (mus[i - 1] * mus[i]) ** 0.5
            mus[i:i] = [mu]
            cvs[i:i] = [self._calc_cv_score(mu, X, y, weights, k)]

        self.mus = mus
        self.cvs = cvs
        return mus[np.nanargmax(cvs)]


def l1regls(A, b):
    """
    Returns the solution of l1-norm regularized least-squares problem
        minimize || A*x - b ||_2^2  + || x ||_1.
    """

    m, n = A.size
    q = matrix(1.0, (2*n, 1))
    q[:n] = -2.0*A.T*b

    def P(u, v, alpha=1.0, beta=0.0):
        """
            v := alpha * 2.0 * [ A'*A, 0; 0, 0 ] * u + beta * v
        """
        v *= beta
        v[:n] += alpha*2.0*A.T*(A*u[:n])

    def G(u, v, alpha=1.0, beta=0.0, trans='N'):
        """
            v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
        """

        v *= beta
        v[:n] += alpha*(u[:n] - u[n:])
        v[n:] += alpha*(-u[:n] - u[n:])

    h = matrix(0.0, (2*n, 1))

    # Customized solver for the KKT system
    #
    #     [  2.0*A'*A  0    I      -I     ] [x[:n] ]     [bx[:n] ]
    #     [  0         0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
    #     [  I        -I   -D1^-1   0     ] [zl[:n]]     [bzl[:n]]
    #     [ -I        -I    0      -D2^-1 ] [zl[n:]]     [bzl[n:]]
    #
    # where D1 = W['di'][:n]**2, D2 = W['di'][:n]**2.
    #
    # We first eliminate zl and x[n:]:
    #
    #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] =
    #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] +
    #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
    #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:]
    #
    #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
    #         - (D2-D1)*(D1+D2)^-1 * x[:n]
    #
    #     zl[:n] = D1 * ( x[:n] - x[n:] - bzl[:n] )
    #     zl[n:] = D2 * (-x[:n] - x[n:] - bzl[n:] ).
    #
    # The first equation has the form
    #
    #     (A'*A + D)*x[:n]  =  rhs
    #
    # and is equivalent to
    #
    #     [ D    A' ] [ x:n] ]  = [ rhs ]
    #     [ A   -I  ] [ v    ]    [ 0   ].
    #
    # It can be solved as
    #
    #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
    #     x[:n] = D^-1 * ( rhs - A'*v ).

    S = matrix(0.0, (m, m))
    # Asc = matrix(0.0, (m, n))
    v = matrix(0.0, (m, 1))

    def Fkkt(W):
        # Factor
        #
        #     S = A*D^-1*A' + I
        #
        # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**-2, D2 = d[n:]**-2.

        d1, d2 = W['di'][:n]**2, W['di'][n:]**2

        # ds is square root of diagonal of D
        ds = math.sqrt(2.0)*div(mul(W['di'][:n], W['di'][n:]), sqrt(d1 + d2))
        d3 = div(d2 - d1, d1 + d2)

        # Asc = A*diag(d)^-1/2
        Asc = A * spdiag(ds**-1)

        # S = I + A * D^-1 * A'
        blas.syrk(Asc, S)
        S[::m+1] += 1.0
        lapack.potrf(S)

        def g(x, y, z):
            x[:n] = 0.5 * (x[:n] - mul(d3, x[n:]) +
                           mul(d1, z[:n] + mul(d3, z[:n])) -
                           mul(d2, z[n:] - mul(d3, z[n:])))
            x[:n] = div(x[:n], ds)

            # Solve
            #
            #     S * v = 0.5 * A * D^-1 * ( bx[:n] -
            #         (D2-D1)*(D1+D2)^-1 * bx[n:] +
            #         D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bzl[:n] -
            #         D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bzl[n:] )

            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)

            # x[:n] = D^-1 * ( rhs - A'*v ).
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
            x[:n] = div(x[:n], ds)

            # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bzl[:n]  - D2*bzl[n:] )
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]
            x[n:] = div(x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1+d2)\
                - mul(d3, x[:n])

            # zl[:n] = D1^1/2 * (  x[:n] - x[n:] - bzl[:n] )
            # zl[n:] = D2^1/2 * ( -x[:n] - x[n:] - bzl[n:] ).
            z[:n] = mul(W['di'][:n], x[:n] - x[n:] - z[:n])
            z[n:] = mul(W['di'][n:], -x[:n] - x[n:] - z[n:])

        return g

    return solvers.coneqp(P, q, G, h, kktsolver=Fkkt)['x'][:n]


import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

# TODO in all adaptive versions solve problem once before loop and change w_prev order
# TODO write clean CVX estimator class and subclasses for each estimator


def adaptive_lasso(X, y, alpha=1.0, max_iter=5, epsilon=None, tol=1E-8):
    epsilon = epsilon or np.finfo(float).eps
    n, m = X.shape
    w = cp.Parameter(shape=m, nonneg=True)
    # alpha = cp.Parameter(nonneg=True)
    beta = cp.Variable(m)
    objective = cp.sum_squares(X @ beta - y) + alpha * cp.norm1(cp.multiply(w, beta))
    problem = cp.Problem(cp.Minimize(objective))
    w_prev = np.ones(m)
    w.value = w_prev
    for _ in range(max_iter):
        problem.solve(warm_start=True)
        w.value = 1.0/(abs(beta.value) + epsilon)
        if np.linalg.norm(w.value - w_prev) <= tol:
            break
        w_prev = w.value
    return beta.value

def adaptive_lasso_sklearn(X, y, alpha=1.0, max_iter=5, epsilon=None, tol=1E-8):
    epsilon = epsilon or np.finfo(float).eps
    n, m = X.shape
    w = np.ones(m)
    est = Lasso(alpha=alpha, fit_intercept=True, warm_start=True)
    w_prev = w
    for _ in range(max_iter):
        X_w = X/w[np.newaxis, :]
        est.fit(X_w, y)
        w = 1.0/ (2. * np.sqrt(np.abs(est.coef_)) + epsilon)
        if np.linalg.norm(w - w_prev) <= tol:
            break
        w_prev = w
    return est.coef_ / w_prev

def group_lasso(X, y, groups, alpha):
    groups = np.asarray(groups)
    group_inds = np.unique(groups)
    sizes = np.sqrt([sum(groups == i) for i in group_inds]) #  if i != -1
    beta = cp.Variable(X.shape[1])
    grp_reg = cp.hstack(cp.norm2(beta[groups == i]) for i in group_inds)
    objective = cp.sum_squares(X @ beta - y) + alpha * (sizes @ grp_reg)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()
    return beta.value


def sparse_group_lasso(X, y, groups, alpha, l1_ratio=0.5):
    groups = np.asarray(groups)
    group_inds = np.unique(groups)
    sizes = np.sqrt([sum(groups == i) for i in group_inds]) #  if i != -1
    #betas = [cp.Variable(size) for size in sizes] # if using -1 to ignore need to add betas for singletons
    #loss = sum(X[:, groups == gid] @ beta for gid, beta in zip(group_inds, betas))
    #group_reg = sum(cp.sqrt(size) * cp.norm2(beta) for size, beta in zip(sizes, betas))
    #l1_reg = cp.norm1(cp.hstack(betas))
    beta = cp.Variable(X.shape[1])
    grp_reg = sum(size * cp.norm2(beta[groups == gid])
                  for size, gid in zip(sizes, group_inds))
    l1_reg = cp.norm1(beta)
    lambd1, lambd2 = l1_ratio * alpha, (1 - l1_ratio) *  alpha
    objective = cp.sum_squares(X @ beta - y) + lambd1 * l1_reg + lambd2 * grp_reg
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()
    #inv = np.concatenate([np.where(groups == gid)[0] for gid in group_inds])
    #beta = np.concatenate([beta.value for beta in betas])
    return beta.value


def adaptive_group_lasso(X, y, groups, alpha=1.0, max_iter=5, epsilon=None, tol=1E-8):
    epsilon = epsilon or np.finfo(float).eps
    groups = np.asarray(groups)
    group_inds = np.unique(groups)
    sizes = np.sqrt([sum(groups == i) for i in group_inds]) #  if i != -1
    w = cp.Parameter(shape=len(sizes), nonneg=True)
    beta = cp.Variable(X.shape[1])
    grp_reg = w @ cp.hstack(cp.norm2(beta[groups == i]) for i in group_inds)
    objective = cp.sum_squares(X @ beta - y) + alpha * grp_reg
    problem = cp.Problem(cp.Minimize(objective))
    w_prev = sizes
    w.value = w_prev
    for _ in range(max_iter):
        problem.solve(warm_start=True)
        grp_norm = np.array([np.linalg.norm(beta.value[groups == gid]) for gid in group_inds])
        w.value = sizes/(grp_norm + epsilon)
        if np.linalg.norm(w.value - w_prev) <= tol:
            break
        w_prev = w.value
    return beta.value


def adaptive_sparse_group_lasso(X, y, groups, l1_ratio=0.5, alpha=1.0,
                                max_iter=5, epsilon=None, weights='lasso',
                                tol=1E-8):
    # TODO allow adaptive weights on lasso/group/both
    epsilon = epsilon or np.finfo(float).eps
    groups = np.asarray(groups)
    group_inds = np.unique(groups)
    sizes = np.sqrt([sum(groups == i) for i in group_inds]) #  if i != -1
    w_grp = cp.Parameter(shape=len(sizes), nonneg=True)
    w_l1 = cp.Parameter(X.shape[1], nonneg=True)
    beta = cp.Variable(X.shape[1])
    grp_reg = w_grp @ cp.hstack(cp.norm2(beta[groups == i]) for i in group_inds)
    l1_reg = cp.norm1(cp.multiply(w_l1, beta))
    lmb_l1, lmb_grp = l1_ratio * alpha, (1 - l1_ratio) *  alpha
    objective = cp.sum_squares(X @ beta - y) + lmb_l1 * l1_reg + lmb_grp * grp_reg
    problem = cp.Problem(cp.Minimize(objective))
    w_gprev = sizes
    w_l1prev = np.ones(X.shape[1])
    w_grp.value = w_gprev
    w_l1.value = w_l1prev
    for _ in range(max_iter):
        problem.solve(warm_start=True)
        grp_norm = np.array([np.linalg.norm(beta.value[groups == gid]) for gid in group_inds])
        w_grp.value = sizes/(grp_norm + epsilon)
        w_l1.value = 1.0/(abs(beta.value) + epsilon)
        if np.linalg.norm(w_grp.value - w_gprev) <= tol and np.linalg.norm(w_l1.value - w_l1prev):
            break
        w_gprev = w_grp.value
        w_l1prev = w_l1.value
    return beta.value


X, y, coef = make_regression(n_samples=306, n_features=1000, n_informative=60,
                             noise=0.1, shuffle=True, coef=True, random_state=30)
X.shape
y.shape
coef_ = adaptive_lasso(X, y, epsilon=1E-10, max_iter=3)
coef_sk = adaptive_lasso_sklearn(X, y, epsilon=1E-10, max_iter=1000)
sum(abs(coef) > 0)
sum(abs(coef_) > 1E-10)
sum(abs(coef_sk) > 1E-10)

sum((coef - coef_)**2)
sum((coef - coef_sk)**2)

fig, ax = plt.subplots()
ax.plot(coef, 's')
ax.plot(coef_, 'o')
ax.plot(coef_sk, '*')
fig

groups = np.random.randint(0, 500, size=len(coef_))
coef_sgl = sparse_group_lasso(X, y, groups, alpha=1.0, l1_ratio=0.95)
coef_gl = group_lasso(X, y, groups, alpha=1.0)
coef_agl = adaptive_group_lasso(X, y, groups, epsilon=1E-5, max_iter=1)
coef_asgl = adaptive_sparse_group_lasso(X, y, groups, l1_ratio=0.1, max_iter=5, epsilon=1E-5)
fig, ax = plt.subplots()
ax.plot(coef, 'o')
ax.plot(coef_sgl, '*')
ax.plot(coef_gl, '^')
ax.plot(coef_agl, 'v')
ax.plot(coef_asgl, '.')

fig
sum(abs(coef_sgl) > 1E-2)
sum(abs(coef_agl) > 1E-10)
sum(abs(coef_) > 1E-10)
sum(abs(coef_asgl) > 1E-10)
