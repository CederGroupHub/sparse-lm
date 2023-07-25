"""
=======================================
Tuning hyperparameters with line search
=======================================

Line search can typically be used in optimizing regressors with multiple weakly or
uncorrelated hyperparameters.

This example also showcases the usage of mixed L0 regressor where using a standard
grid search can be too computationally expensive..
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from sparselm.model import L2L0
from sparselm.model_selection import LineSearchCV

X, y, coef = make_regression(
    n_samples=60,
    n_features=30,
    n_informative=8,
    noise=40.0,
    bias=-15.0,
    coef=True,
    random_state=0,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# create an l2l0 estimator.
# Groups for parameters must be provided each coefficient is in a singleton group.
groups = np.arange(30, dtype=int)
l2l0 = L2L0(groups, fit_intercept=True, solver="GUROBI", solver_options={"Threads": 4})

# create cv search objects for each estimator
cv5 = KFold(n_splits=5, shuffle=True, random_state=0)
# LineSearchCV requires the parameters grid to be provided in a list of tuple format,
# with order of parameters in the list being the order of them getting searched per
# iteration.
# The following example specifies the parameter alpha to be scanned first, then the
# parameter eta.
params = [("alpha", np.logspace(-6, 1, 5)), ("eta", np.logspace(-7, -1, 5))]

l2l0_cv = LineSearchCV(l2l0, params, cv=cv5, n_jobs=4)

# fit models on training data
l2l0_cv.fit(X_train, y_train)

# calculate model performance on test and train data
l2l0_train = {
    "r2": r2_score(y_train, l2l0_cv.predict(X_train)),
    "rmse": np.sqrt(mean_squared_error(y_train, l2l0_cv.predict(X_train))),
}

l2l0_test = {
    "r2": r2_score(y_test, l2l0_cv.predict(X_test)),
    "rmse": np.sqrt(mean_squared_error(y_test, l2l0_cv.predict(X_test))),
}

print("Performance metrics:")
print(f"    train r2: {l2l0_train['r2']:.3f}")
print(f"    test r2: {l2l0_test['r2']:.3f}")
print(f"    train rmse: {l2l0_train['rmse']:.3f}")
print(f"    test rmse: {l2l0_test['rmse']:.3f}")
