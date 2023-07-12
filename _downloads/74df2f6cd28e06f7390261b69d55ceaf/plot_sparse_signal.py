"""
=========================
Recovering sparse signals
=========================

In this example we compare the results obtained from `BestSubsetSelection` with
those obtained using the `OrthogonalMatchingPursuit` regressor from **scikit-learn**.

Note that although using best subset selection tend to give more accurate results,
`OrthogonalMatchingPursuit` scales much better to larger problems.

This example is adapted from the scikit-learn documentation:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_omp.html#sphx-glr-auto-examples-linear-model-plot-omp-py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit

from sparselm.model import BestSubsetSelection

n_components, n_features = 50, 20
n_nonzero_coefs = 8

# generate the data
y, X, w = make_sparse_coded_signal(
    n_samples=1,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)
X = X.T
(idx,) = w.nonzero()

# distort the clean signal
y_noisy = y + 0.005 * np.random.randn(len(y))

# plot the sparse signal
plt.figure(figsize=(14, 7))
plt.subplot(3, 2, (1, 2))
plt.xlim(0, n_components)
plt.title("Sparse signal")
plt.stem(idx, w[idx])

# plot the noise-free reconstruction
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
omp.fit(X, y)
coef = omp.coef_
(idx_r,) = coef.nonzero()
plt.subplot(3, 2, 3)
plt.xlim(0, n_components)
plt.title("Orthogonal Matching Pursuit (noise-free measurements)")
plt.stem(idx_r, coef[idx_r])

bss = BestSubsetSelection(
    sparse_bound=n_nonzero_coefs, solver="GUROBI", solver_options={"Threads": 8}
)
bss.fit(X, y)
coef = bss.coef_
(idx_r,) = coef.nonzero()
plt.subplot(3, 2, 4)
plt.xlim(0, n_components)
plt.title("Best Subset Selection (noise-free measurements)")
plt.stem(idx_r, coef[idx_r])

# plot the noisy reconstruction
omp.fit(X, y_noisy)
coef = omp.coef_
(idx_r,) = coef.nonzero()
plt.subplot(3, 2, 5)
plt.xlim(0, n_components)
plt.title("Orthogonal Matching Pursuit recovery (noisy measurements)")
plt.stem(idx_r, coef[idx_r])

bss.fit(X, y_noisy)
coef = bss.coef_
(idx_r,) = coef.nonzero()
plt.subplot(3, 2, 6)
plt.xlim(0, n_components)
plt.title("Best Subset Selection (noisy measurements)")
plt.stem(idx_r, coef[idx_r])

plt.tight_layout()
plt.show()
