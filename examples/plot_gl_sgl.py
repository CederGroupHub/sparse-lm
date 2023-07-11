"""
=========================
(Sparse) Group regression
=========================

This examples shows how to use group lasso and sparse group lasso to fit a simulated
dataset with group-level sparsity and within-group sparsity.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from sparselm.dataset import make_group_regression
from sparselm.model import GroupLasso, SparseGroupLasso

warnings.filterwarnings("ignore", category=UserWarning)  # ignore convergence warnings

# generate a dataset with group-level sparsity only
X, y, groups, coefs = make_group_regression(
    n_samples=400,
    n_groups=10,
    n_features_per_group=10,
    n_informative_groups=5,
    frac_informative_in_group=1.0,
    bias=-10.0,
    noise=200.0,
    coef=True,
    random_state=0,
)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create estimators
cv5 = KFold(n_splits=5, shuffle=True, random_state=0)
lasso_cv = GridSearchCV(
    Lasso(fit_intercept=True), {"alpha": np.logspace(0, 2, 5)}, cv=cv5, n_jobs=-1
)
lasso_cv.fit(X_train, y_train)
glasso_cv = GridSearchCV(
    GroupLasso(groups=groups, fit_intercept=True),
    {"alpha": np.logspace(0, 2, 5)},
    cv=cv5,
    n_jobs=-1,
)
glasso_cv.fit(X_train, y_train)

# Plot predicted values
fig, ax = plt.subplots()
ax.plot(
    y_test, glasso_cv.predict(X_test), marker="o", ls="", alpha=0.5, label="group lasso"
)
ax.plot(y_test, lasso_cv.predict(X_test), marker="o", ls="", alpha=0.5, label="lasso")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
ax.legend()
ax.set_xlabel("true values")
ax.set_ylabel("predicted values")
fig.show()

# calculate model performance on test and train data
lasso_train = {
    "r2": r2_score(y_train, lasso_cv.predict(X_train)),
    "rmse": np.sqrt(mean_squared_error(y_train, lasso_cv.predict(X_train))),
}

lasso_test = {
    "r2": r2_score(y_test, lasso_cv.predict(X_test)),
    "rmse": np.sqrt(mean_squared_error(y_test, lasso_cv.predict(X_test))),
}

glasso_train = {
    "r2": r2_score(y_train, glasso_cv.predict(X_train)),
    "rmse": np.sqrt(mean_squared_error(y_train, glasso_cv.predict(X_train))),
}

glasso_test = {
    "r2": r2_score(y_test, glasso_cv.predict(X_test)),
    "rmse": np.sqrt(mean_squared_error(y_test, glasso_cv.predict(X_test))),
}

print("------- Performance metrics for signal with group-level sparsity only -------\n")

print("Lasso performance metrics:")
print(f"    train r2: {lasso_train['r2']:.3f}")
print(f"    test r2: {lasso_test['r2']:.3f}")
print(f"    train rmse: {lasso_train['rmse']:.3f}")
print(f"    test rmse: {lasso_test['rmse']:.3f}")

print("Group Lasso performance metrics:")
print(f"    train r2: {glasso_train['r2']:.3f}")
print(f"    test r2: {glasso_test['r2']:.3f}")
print(f"    train rmse: {glasso_train['rmse']:.3f}")
print(f"    test rmse: {glasso_test['rmse']:.3f}")

# generate a dataset with group-level sparsity and within-group sparsity
X, y, groups, coefs = make_group_regression(
    n_samples=400,
    n_groups=10,
    n_features_per_group=10,
    n_informative_groups=5,
    frac_informative_in_group=0.5,
    bias=-10.0,
    noise=100.0,
    coef=True,
    random_state=0,
)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

glasso_cv = GridSearchCV(
    GroupLasso(groups=groups, fit_intercept=True),
    {"alpha": np.logspace(0, 2, 5)},
    cv=cv5,
    n_jobs=-1,
)
sglasso_cv = GridSearchCV(
    SparseGroupLasso(groups=groups, fit_intercept=True),
    {"alpha": np.logspace(0, 2, 5), "l1_ratio": np.arange(0.3, 0.8, 0.1)},
    cv=cv5,
    n_jobs=-1,
)
glasso_cv.fit(X_train, y_train)
sglasso_cv.fit(X_train, y_train)

# Plot predicted values
fig, ax = plt.subplots()
ax.plot(
    y_test, glasso_cv.predict(X_test), marker="o", ls="", alpha=0.5, label="group lasso"
)
ax.plot(
    y_test,
    sglasso_cv.predict(X_test),
    marker="o",
    ls="",
    alpha=0.5,
    label="sparse group lasso",
)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
ax.legend()
ax.set_xlabel("true values")
ax.set_ylabel("predicted values")
fig.show()

# calculate model performance on test and train data
glasso_train = {
    "r2": r2_score(y_train, glasso_cv.predict(X_train)),
    "rmse": np.sqrt(mean_squared_error(y_train, glasso_cv.predict(X_train))),
}

glasso_test = {
    "r2": r2_score(y_test, glasso_cv.predict(X_test)),
    "rmse": np.sqrt(mean_squared_error(y_test, glasso_cv.predict(X_test))),
}

sglasso_train = {
    "r2": r2_score(y_train, sglasso_cv.predict(X_train)),
    "rmse": np.sqrt(mean_squared_error(y_train, sglasso_cv.predict(X_train))),
}

sglasso_test = {
    "r2": r2_score(y_test, sglasso_cv.predict(X_test)),
    "rmse": np.sqrt(mean_squared_error(y_test, sglasso_cv.predict(X_test))),
}


print(
    "------- Performance metrics for signal with group and within group sparsity -------\n"
)

print("Group Lasso performance metrics:")
print(f"    train r2: {glasso_train['r2']:.3f}")
print(f"    test r2: {glasso_test['r2']:.3f}")
print(f"    train rmse: {glasso_train['rmse']:.3f}")
print(f"    test rmse: {glasso_test['rmse']:.3f}")

print("Sparse Group Lasso performance metrics:")
print(f"    train r2: {sglasso_train['r2']:.3f}")
print(f"    test r2: {sglasso_test['r2']:.3f}")
print(f"    train rmse: {sglasso_train['rmse']:.3f}")
print(f"    test rmse: {sglasso_test['rmse']:.3f}")
