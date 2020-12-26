"""A set of generalized lasso estimators.

Estimators follow scikit-learn interface, but use cvxpy to set up and solver
optimization problem.
"""

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


