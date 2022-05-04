"""A set of generalized lasso estimators.
* Lasso
* Group Lasso
* Overlap Group Lasso
* Sparse Group Lasso

Estimators follow scikit-learn interface, but use cvxpy to set up and solve
optimization problem.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

import warnings
import cvxpy as cp
import numpy as np

from sparselm.model.base import CVXEstimator


class Lasso(CVXEstimator):
    """
    Lasso Estimator implemented with cvxpy.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * ||Beta||_1
    Where w represents a vector of weights that is iteratively updated.
    """

    def __init__(self, alpha=1.0, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """
        Args:
            alpha (float):
                Regularization hyper-parameter.
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
        self._alpha = cp.Parameter(value=alpha, nonneg=True)
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         copy_X=copy_X, warm_start=warm_start, solver=solver,
                         **kwargs)

    @property
    def alpha(self):
        return self._alpha.value

    @alpha.setter
    def alpha(self, val):
        self._alpha.value = val

    def _gen_regularization(self, X):
        return self._alpha * cp.norm1(self._beta)

    def _gen_objective(self, X, y):
        # can also use cp.norm2(X @ self._beta - y)**2 not sure whats better
        reg = self._gen_regularization(X)
        objective = 1 / (2 * X.shape[0]) * cp.sum_squares(X @ self._beta - y) \
            + reg
        return objective


class GroupLasso(Lasso):
    """Group Lasso implementation.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * \sum_{G} w_G * ||Beta_G||_2
    Where G represents groups of features/coefficients
    """

    def __init__(self, groups, alpha=1.0, group_weights=None,
                 standardize=False, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to.
            alpha (float):
                Regularization hyper-parameter.
            fit_intercept (bool):
                Whether the intercept should be estimated or not.
                If False, the data is assumed to be already centered.
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the groups given. If you need all groups
                weighted equally just pass an array of ones.
            standardize (bool): optional
                Whether to standardize the group regularization penalty using
                the feature matrix. See the following for reference:
                http://faculty.washington.edu/nrsimon/standGL.pdf
            normalize (bool): optional
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
                See docs linked in CVXEstimator base class for more info.
        """
        self.groups = np.asarray(groups)
        self.group_masks = [self.groups == i for i in np.unique(groups)]
        self.standardize = standardize
        self._group_norms = None

        if group_weights is not None:
            if len(group_weights) != len(self.group_masks):
                raise ValueError(
                    'group_weights must be the same length as the number of '
                    f'groups:  {len(group_weights)} != {len(self.group_masks)}')
        self.group_weights = group_weights if group_weights is not None else \
            np.sqrt([sum(mask) for mask in self.group_masks])

        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

    def _gen_group_norms(self, X):
        if self.standardize:
            grp_norms = cp.hstack(
                [cp.norm2(X[:, mask] @ self._beta[mask])
                 for mask in self.group_masks]
            )
        else:
            grp_norms = cp.hstack(
                [cp.norm2(self._beta[mask]) for mask in self.group_masks])
        self._group_norms = grp_norms
        return grp_norms

    def _gen_regularization(self, X):
        return self._alpha * (self.group_weights @ self._gen_group_norms(X))


class OverlapGroupLasso(GroupLasso):
    """Overlap Group Lasso implementation.

    Regularized model:
        || X * Beta - y ||^2_2 + alpha * \sum_{G} w_G * ||Beta_G||_2
    Where G represents groups of features/coefficients, and overlaping groups
    are acceptable. Meaning a coefficients can be in more than one group.
    """

    def __init__(self, group_list, alpha=1.0, group_weights=None,
                 standardize=False, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """Initialize estimator.

        Args:
            group_list (list of lists):
                list of lists of integers specifying groups. The length of the
                list holding lists should be the same as model. Each inner list
                has integers specifying the groups the coefficient for that
                index belongs to. i.e. [[1,2],[2,3],[1,2,3]] means the first
                coefficient belongs to group 1 and 2, the second to 2, and 3
                and the third to 1, 2 and 3. In other words the 3 groups would
                be: (0, 2), (0, 1, 2), (1, 2)
            alpha (float):
                Regularization hyper-parameter.
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the number of different groups given.
                If you need all groups weighted equally just pass an array of
                ones.
            standardize (bool): optional
                Whether to standardize the group regularization penalty using
                the feature matrix. See the following for reference:
                http://faculty.washington.edu/nrsimon/standGL.pdf
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
                See docs linked in CVXEstimator base class for more info.
        """
        self.group_list = group_list
        self.group_ids = np.unique([gid for grp in group_list for gid in grp])
        self.group_ids.sort()
        beta_indices = [[i for i, grp in enumerate(group_list) if grp_id in grp]
                        for grp_id in self.group_ids]
        extended_groups = np.concatenate(
            [len(g) * [i, ] for i, g in enumerate(beta_indices)])
        self.beta_indices = np.concatenate(beta_indices)

        super().__init__(
            extended_groups, alpha=alpha, group_weights=group_weights,
            standardize=standardize, fit_intercept=fit_intercept,
            normalize=normalize, copy_X=copy_X, warm_start=warm_start,
            solver=solver, **kwargs)

    def _solve(self, X, y, *args, **kwargs):
        """Solve the cvxpy problem."""
        problem = self._get_problem(X[:, self.beta_indices], y)
        problem.solve(solver=self.solver, warm_start=self.warm_start,
                      **self.solver_opts)
        beta = np.array(
            [sum(self._beta.value[self.beta_indices == i])
             for i in range(X.shape[1])]
        )
        return beta


class SparseGroupLasso(GroupLasso):
    """Sparse Group Lasso.

    Regularized model:
        || X * Beta - y ||^2_2
            + alpha * l1_ratio * ||Beta||_1
            + alpha * (1 - l1_ratio) * \sum_{G}||Beta_G||_2
    Where G represents groups of features / coefficients
    """

    def __init__(self, groups, l1_ratio=0.5, alpha=1.0, group_weights=None,
                 standardize=False, fit_intercept=False, normalize=False,
                 copy_X=True, warm_start=False, solver=None, **kwargs):
        """Initialize estimator.

        Args:
            groups (list or ndarray):
                array-like of integers specifying groups. Length should be the
                same as model, where each integer entry specifies the group
                each parameter corresponds to.
            l1_ratio (float):
                Mixing parameter between l1 and group lasso regularization.
            alpha (float):
                Regularization hyper-parameter.
            group_weights (ndarray): optional
                Weights for each group to use in the regularization term.
                The default is to use the sqrt of the group sizes, however any
                weight can be specified. The array must be the
                same length as the groups given. If you need all groups
                weighted equally just pass an array of ones.
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
                See docs linked in CVXEstimator base class for more info.
        """
        super().__init__(groups=groups, alpha=alpha,
                         group_weights=group_weights,
                         standardize=standardize,
                         fit_intercept=fit_intercept,
                         normalize=normalize, copy_X=copy_X,
                         warm_start=warm_start, solver=solver, **kwargs)

        if not 0 <= l1_ratio <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        elif l1_ratio == 0.0:
            warnings.warn(
                'It is more efficient to use GroupLasso directly than '
                'SparseGroupLasso with l1_ratio=0', UserWarning)
        elif l1_ratio == 1.0:
            warnings.warn(
                'It is more efficient to use Lasso directly than '
                'SparseGroupLasso with l1_ratio=1', UserWarning)

        self._lambda1 = cp.Parameter(nonneg=True, value=l1_ratio * alpha)
        self._lambda2 = cp.Parameter(nonneg=True, value=(1 - l1_ratio) * alpha)
        # save exact value so sklearn clone is happy dappy
        self._l1_ratio = l1_ratio

    @Lasso.alpha.setter
    def alpha(self, val):
        self._alpha.value = val
        self._lambda1.value = self.l1_ratio * val
        self._lambda2.value = (1 - self.l1_ratio) * val

    @property
    def l1_ratio(self):
        return self._l1_ratio

    @l1_ratio.setter
    def l1_ratio(self, val):
        if not 0 <= val <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        self._l1_ratio = val
        self._lambda1.value = val * self.alpha
        self._lambda2.value = (1 - val) * self.alpha

    def _gen_regularization(self, X):
        grp_norms = super()._gen_group_norms(X)
        l1_reg = cp.norm1(self._beta)
        reg = self._lambda1 * l1_reg + \
            self._lambda2 * (self.group_weights @ grp_norms)
        return reg
