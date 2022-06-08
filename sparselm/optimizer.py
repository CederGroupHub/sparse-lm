"""Classes implementing parameters selection beyond GridsearchCV."""
from abc import ABCMeta
import re
import numbers
import numpy as np
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid


class BaseSearchOneStd(BaseSearchCV, metaclass=ABCMeta):
    """Abstract base class for hyper-parameter search with cross-validation."""
    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Rewrite to implement one standard error rule."""
        if callable(refit):
            # If callable, refit is expected to return the index of the best
            # parameter set.
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError("best_index_ returned is not an integer")
            if best_index < 0 or best_index >= len(results["params"]):
                raise IndexError("best_index_ index out of range")
        else:
            # Use rank_test_metric because some score like r2 needs to be
            # maximized, not minimized.
            opt_index = results[f"rank_test_{refit_metric}"].argmin()
            m = results[f"mean_test_{refit_metric}"][opt_index]
            sig = results[f"std_test_{refit_metric}"][opt_index]
            metrics = results[f"mean_test_{refit_metric}"]
            param_names = [key for key in results if re.match(r"^param_(\w+)", key)]
            params = []
            # Will only apply one std rule on numerical, all positive parameters,
            # which are usually regularization factors.
            # All parameters are equally treated in grid and line search.
            for name in param_names:
                if all(isinstance(val, numbers.Number) for val in results[name]):
                    p = np.array(results[name], dtype=float)
                    if np.all(p > -1E-6):
                        params.append(p)
            params_sum = np.sum(params, axis=0)
            best_index = np.argmax(params_sum - (metrics - m + sig) ** 2)

        return best_index


class GridSearchOneStd(BaseSearchOneStd):
    """Exhaustive search over specified parameter values for an estimator.

    Same as GridSearchCV but we apply one standard error rule on all
    non-negative numerical hyperparameters, in order to get a
    robust sparce estimation. Same documentation as scikit-learn's
    GridSearchCV.
    """

    _required_parameters = ["estimator", "param_grid"]

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))



