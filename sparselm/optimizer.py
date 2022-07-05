"""Classes implementing parameters selection beyond GridsearchCV."""
import re
import numbers
import numpy as np
import warnings
from copy import deepcopy

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._search import BaseSearchCV


class GridSearch(GridSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Same as GridSearchCV but we allow one standard error rule on all
    non-negative numerical hyper-parameters, in order to get a
    robust sparce estimation. Same documentation as scikit-learn's
    GridSearchCV.

    An additional class variable opt_selection named opt_selection
    is added to allow switching hyper params selection mode. Currently,
    supports "max_r2" (default), which is to maximize the r2 score;
    also supports "one_std_r2", which is to apply one-standard-error
    rule the r2 score.
    """
    opt_selection = "max_r2"

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Rewritten to implement one standard error rule."""
        if callable(refit) or GridSearch.opt_selection == "max_r2":
            # Use maximize r2.
            return GridSearchCV._select_best_index(refit, refit_metric, results)
        elif GridSearch.opt_selection == "one_std_r2":
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
                    if np.all(p > -1E-9):
                        params.append(p)
            params_sum = np.sum(params, axis=0)
            return np.argmax(params_sum - (metrics - m + sig) ** 2)
        else:
            raise ValueError(f"{GridSearch.opt_selection} not supported!")


class LineSearch(BaseSearchCV):
    """Implements line search.

    In line search, we do 1 dimensional grid searches on each hyper-param up
    to a certain number of iterations. Each search will generate a GridSearchCV
    object.
    """
    # Class default optimal selection method for all hyper-params dimensions.
    opt_selection = "max_r2"

    def __init__(
            self,
            estimator,
            param_grid,
            *,
            opt_selection_methods=None,
            n_iter=None,
            scoring=None,
            n_jobs=None,
            refit=True,
            cv=None,
            verbose=0,
            pre_dispatch="2*n_jobs",
            error_score=np.nan,
            return_train_score=False,
    ):
        """
        Args:
            estimator(Estimator):
                A object of that type is instantiated for each grid point.
                This is assumed to implement the scikit-learn estimator interface.
                Either estimator needs to provide a ``score`` function,
                or ``scoring`` must be passed.
            param_grid(list[tuple] or list[list[tuple]]):
                List of tuples with parameters names (`str`) as first element
                and lists of parameter settings to try as the second element;
                or a list of such lists, in which case the grids spanned by each
                sub-list in the list are explored.
                In LineSearch, the hyper-params given first will be searched first
                in a cycle.
            opt_selection_methods(list(str) or str, default=None):
                The method to select optimal hyper params. Default to "max_r2", which
                means to maximize r2 score. Can also choose "one_std_r2", which means
                to apply one standard error rule on r2 scores. If given as a list of str,
                it allows different selection method for corresponding hyper-params in
                the param_grid argument.
            n_iter(int, default=None):
                Number of iterations to perform. One iteration means a 1D search on
                one hyper-param, and we scan one hyper-param at a time in the order of
                param_grid.
                n_iter must be at least as large as the number of hyper-params. Default
                is 2 * number of hyper-params.
            scoring(str, callable, list, tuple or dict, default=None):
                Strategy to evaluate the performance of the cross-validated model on
                the test set.
                If `scoring` represents a single score, one can use:
                - a single string (see :ref:`scoring_parameter`);
                - a callable (see :ref:`scoring`) that returns a single value.
                If `scoring` represents multiple scores, one can use:
                - a list or tuple of unique strings;
                - a callable returning a dictionary where the keys are the metric
                  names and the values are the metric scores;
                - a dictionary with metric names as keys and callables a values.
                See :ref:`multimetric_grid_search` for an example.
            n_jobs(int, default=None):
                Number of jobs to run in parallel.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.
            refit(bool, str, or callable, default=True)
                Refit an estimator using the best found parameters on the whole
                dataset.
                For multiple metric evaluation, this needs to be a `str` denoting the
                scorer that would be used to find the best parameters for refitting
                the estimator at the end.
                Where there are considerations other than maximum score in
                choosing a best estimator, ``refit`` can be set to a function which
                returns the selected ``best_index_`` given ``cv_results_``. In that
                case, the ``best_estimator_`` and ``best_params_`` will be set
                according to the returned ``best_index_`` while the ``best_score_``
                attribute will not be available.
                The refitted estimator is made available at the ``best_estimator_``
                attribute and permits using ``predict`` directly on this
                instance.
                Also for multiple metric evaluation, the attributes ``best_index_``,
                ``best_score_`` and ``best_params_`` will only be available if
                ``refit`` is set and all of them will be determined w.r.t this specific
                scorer.
                See ``scoring`` parameter to know more about multiple metric
                evaluation.
            cv(int, cross-validation generator or an iterable, default=None):
                Determines the cross-validation splitting strategy.
                Possible inputs for cv are:
                - None, to use the default 5-fold cross validation,
                - integer, to specify the number of folds in a `(Stratified)KFold`,
                - :term:`CV splitter`,
                - An iterable yielding (train, test) splits as arrays of indices.
                For integer/None inputs, if the estimator is a classifier and ``y`` is
                either binary or multiclass, :class:`StratifiedKFold` is used. In all
                other cases, :class:`KFold` is used. These splitters are instantiated
                with `shuffle=False` so the splits will be the same across calls.
                Refer :ref:`User Guide <cross_validation>` for the various
                cross-validation strategies that can be used here.
            verbose(int, default=0):
                Controls the verbosity: the higher, the more messages.
                - >1 : the computation time for each fold and parameter candidate is
                  displayed;
                - >2 : the score is also displayed;
                - >3 : the fold and candidate parameter indexes are also displayed
                  together with the starting time of the computation.
            pre_dispatch(int, or str, default='2*n_jobs'):
                Controls the number of jobs that get dispatched during parallel
                execution. Reducing this number can be useful to avoid an
                explosion of memory consumption when more jobs get dispatched
                than CPUs can process. This parameter can be:
                    - None, in which case all the jobs are immediately
                      created and spawned. Use this for lightweight and
                      fast-running jobs, to avoid delays due to on-demand
                      spawning of the jobs
                    - An int, giving the exact number of total jobs that are
                      spawned
                    - A str, giving an expression as a function of n_jobs,
                      as in '2*n_jobs'
            error_score('raise' or numeric, default=np.nan):
                Value to assign to the score if an error occurs in estimator fitting.
                If set to 'raise', the error is raised. If a numeric value is given,
                FitFailedWarning is raised. This parameter does not affect the refit
                step, which will always raise the error.
            return_train_score(bool, default=False):
                If ``False``, the ``cv_results_`` attribute will not include training
                scores.
                Computing training scores is used to get insights on how different
                parameter settings impact the overfitting/underfitting trade-off.
                However, computing the scores on the training set can be computationally
                expensive and is not strictly required to select the parameters that
                yield the best generalization performance.
        """
        # These are equally passed into GridSearchCV objects at each iteration.
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

        self.param_grid = param_grid
        if isinstance(param_grid[0][0], str) \
                and isinstance(param_grid[0], (tuple, list)):
            self.n_params = len(param_grid)
        elif isinstance(param_grid[0][0][0], str) \
                and isinstance(param_grid[0][0], (tuple, list)):
            self.n_params = len(param_grid[0])
        else:
            raise ValueError("Parameters grid not given in the correct format!")

        if opt_selection_methods is None:
            self.opt_selection_methods = ["max_r2" for _ in range(self.n_params)]
        elif isinstance(opt_selection_methods, str):
            self.opt_selection_methods = [opt_selection_methods
                                          for _ in range(self.n_params)]
        elif isinstance(opt_selection_methods, (list, tuple)) \
                and isinstance(opt_selection_methods, str) \
                and len(opt_selection_methods) == self.n_params:
            self.opt_selection_methods = opt_selection_methods
        else:
            raise ValueError("Optimal hyperparams selection method"
                             " not given in the correct format!")

        self.n_iter = n_iter if (n_iter is not None and n_iter > 0)\
            else 2 * self.n_params

        # Stores GridSearch object at each iteration
        self._history = []

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Args:
            X(array-like of shape (n_samples, n_features)):
                Training vector, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            y(array-like of shape (n_samples, n_output) or (n_samples,), default=None):
                Target relative to X for classification or regression;
                None for unsupervised learning.
            groups(array-like of shape (n_samples,), default=None):
                Group labels for the samples used while splitting the dataset into
                train/test set. Only used in conjunction with a "Group" :term:`cv`
                instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
            **fit_params:
                Parameters passed to the `fit` method of the estimator.
                If a fit parameter is an array-like whose length is equal to
                `num_samples` then it will be split across CV groups along with `X`
                and `y`. For example, the :term:`sample_weight` parameter is split
                because `len(sample_weights) = len(X)`.
            Returns:
                self(LineSearch):
                    Instance of fitted estimator.
        """
        if len(self._history) > 0:
            warnings.warn("Overwriting existing fit history!")
            self._history = []

        best_line_params_ = None
        for i in range(self.n_iter):
            param_id = i % self.n_params
            if best_line_params_ is None:
                last_params = [values[0] for name, values in self.param_grid]
            else:
                last_params = [best_line_params_[name]
                               for name, values in self.param_grid]

            param_line = {}
            for pid, ((name, values), last_value)\
                    in enumerate(zip(self.param_grid, last_params)):
                param_line[name] = [last_value] if pid != param_id else values

            grid_search = GridSearch(estimator=self.estimator,
                                     param_grid=param_line,
                                     scoring=self.scoring,
                                     n_jobs=self.n_jobs,
                                     refit=self.refit,
                                     cv=self.cv,
                                     verbose=self.verbose,
                                     pre_dispatch=self.pre_dispatch,
                                     error_score=self.error_score,
                                     return_train_score=self.return_train_score)
            grid_search.opt_selection = self.opt_selection_methods[param_id]
            grid_search.fit(X=X, y=y, groups=groups, **fit_params)
            best_line_params_ = deepcopy(grid_search.best_params_)
            self._history.append(grid_search)

        # Buffer fitted attributes into LineSearch object.
        attrs = [v for v in vars(self._history[-1])
                 if v.endswith("_") and not v.startswith("__")]
        for attr in attrs:
            setattr(self, attr, getattr(self._history[-1], attr))
        return self

    def _run_search(self, evaluate_candidates):
        """Muted function, only kept for overwriting abstraction."""
        return
