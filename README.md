<img src="docs/_static/logo.png" width="500px" alt=" ">

Sparse Linear Regression Models
===============================
[![test](https://github.com/CederGroupHub/sparse-lm/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/CederGroupHub/sparse-lm/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/CederGroupHub/sparse-lm/main.svg)](https://results.pre-commit.ci/latest/github/CederGroupHub/sparse-lm/main)
[![pypi version](https://img.shields.io/pypi/v/sparse-lm?color=blue)](https://pypi.org/project/sparse-lm)

> :warning: this package is currently largely lacking in unit-tests.
> Use at your own risk!

**sparse-lm**  includes several regularized regression estimators that are absent in the
`sklearn.linear_model` module. The estimators in **sparse-lm** are designed to fit right into
[scikit-lean](https://scikit-learn.org/stable/index.html) by inheriting from their base
`LinearModel`. But the underlying optimization problem is expressed and solved by
leveraging [cvxpy](https://www.cvxpy.org/).

---------------------------------------------------------------------------------------

Available regression models
---------------------------
- Ordinary Least Squares (`sklearn` may be a better option)
- Lasso (`sklearn` may be a better option)
- Group Lasso, Overlap Group Lasso & Sparse Group Lasso
- Adaptive versions of Lasso, Group Lasso, Overlap Group Lasso, Sparse Group Lasso & Ridged Group Lasso.
- Best Subset Selection, Ridged Best Subset, L0, L1L0 & L2L0 (all with optional grouping of parameters)
  (`gurobi` recommended for performance)

Installation
------------
From pypi:

    pip install sparse-lm

Usage
-----
If you already use **scikit-learn**, using **sparse-lm** will be very easy. Just use any
model like you would any linear model in **scikit-learn**:

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from src.sparselm import AdaptiveLasso

X, y = make_regression(n_samples=200, n_features=5000, random_state=0)
alasso = AdaptiveLasso(fit_intercept=False)
param_grid = {'alpha': np.logsppace(-7, -2)}

cvsearch = GridSearchCV(alasso, param_grid)
cvsearch.fit(X, y)
print(cvsearch.best_params_)
```
