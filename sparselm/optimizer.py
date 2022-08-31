"""Classes implementing parameters selection beyond GridsearchCV."""
__author__ = "Fengyu Xie"

import numbers
import re
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import product

import numpy as np
from joblib import Parallel
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _fit_and_score,
    _insert_error_scores,
    _warn_or_raise_about_fit_failures,
)
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import _check_fit_params, indexable


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

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        opt_selection_method="max_r2",
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        """Initialize CVSearch tool.

        Args:
            estimator(Estimator):
                A object of that type is instantiated for each grid point.
                This is assumed to implement the scikit-learn estimator interface.
                Either estimator needs to provide a ``score`` function,
                or ``scoring`` must be passed.
            param_grid(dict or list[dict]):
                Dictionary representing grid of hyper-parameters with their names
                as keys and possible values. If given as a list of multiple dicts,
                will search on multiple grids in parallel.
            opt_selection_method(str, default=None):
                The method to select optimal hyper params. Default to "max_r2", which
                means to maximize r2 score. Can also choose "one_std_r2", which means
                to apply one standard error rule on r2 scores.
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
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.opt_selection_method = opt_selection_method

    # Provides one-standard-error rule.
    @staticmethod
    def _select_best_index_onestd(refit, refit_metric, results):
        """Rewritten to implement one standard error rule."""
        # Use rank_test_metric because some score like r2 needs to be
        # maximized, not minimized.
        if callable(refit):
            # If callable, refit is expected to return the index of the best
            # parameter set.
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError("best_index_ returned is not an integer")
            if best_index < 0 or best_index >= len(results["params"]):
                raise IndexError("best_index_ index out of range")
        else:
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
                    if np.all(p > -1e-9):
                        params.append(p)
            params_sum = np.sum(params, axis=0)
            one_std_dists = np.abs(metrics - m + sig)
            candidates = np.arange(len(metrics))[
                one_std_dists < (np.min(one_std_dists) + 0.1 * sig)
            ]
            best_index = candidates[np.argmax(params_sum[candidates])]
            return best_index

    # Overwrite original fit method to allow multiple optimal methods.
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
                self(GridSearch):
                    Instance of fitted estimator.
        """
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {} folds for each of {} candidates,"
                        " totalling {} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # Implement more if needed.
            if self.opt_selection_method == "max_r2":
                self.best_index_ = self._select_best_index(
                    self.refit, refit_metric, results
                )
            elif self.opt_selection_method == "one_std_r2":
                self.best_index_ = self._select_best_index_onestd(
                    self.refit, refit_metric, results
                )
            else:
                raise NotImplementedError(
                    f"Method {self.opt_selection_method}" f" not implemented!"
                )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class LineSearch(BaseSearchCV):
    """Implements line search.

    In line search, we do 1 dimensional grid searches on each hyper-param up
    to a certain number of iterations. Each search will generate a GridSearchCV
    object.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        opt_selection_method=None,
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
        """Initialize a LineSearch.

        Args:
            estimator(Estimator):
                A object of that type is instantiated for each grid point.
                This is assumed to implement the scikit-learn estimator interface.
                Either estimator needs to provide a ``score`` function,
                or ``scoring`` must be passed.
            param_grid(list[tuple]):
                List of tuples with parameters names (`str`) as first element
                and lists of parameter settings to try as the second element.
                In LineSearch, the hyper-params given first will be searched first
                in a cycle. Multiple grids search is NOT supported!
            opt_selection_method(list(str) or str, default=None):
                The method to select optimal hyper params. Default to "max_r2", which
                means to maximize r2 score. Can also choose "one_std_r2", which means
                to apply one standard error rule on r2 scores.
                In line search, this argument can also be given as a list of str. This
                will allow different selection methods for corresponding hyper-params in
                the param_grid. For example, a good practice when using L2L0 estimator
                shall be opt_selection_method = ["one_std_r2", "max_r2"] for "alpha"
                and "l0_ratio", respectively.
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
        if isinstance(param_grid[0][0], str) and isinstance(
            param_grid[0], (tuple, list)
        ):
            self.n_params = len(param_grid)
        else:
            raise ValueError("Parameters grid not given in the correct format!")

        if opt_selection_method is None:
            self.opt_selection_methods = ["max_r2" for _ in range(self.n_params)]
        elif isinstance(opt_selection_method, str):
            self.opt_selection_methods = [
                opt_selection_method for _ in range(self.n_params)
            ]
        elif (
            isinstance(opt_selection_method, (list, tuple))
            and all(isinstance(m, str) for m in opt_selection_method)
            and len(opt_selection_method) == self.n_params
        ):
            self.opt_selection_methods = opt_selection_method
        else:
            raise ValueError(
                "Optimal hyperparams selection method"
                " not given in the correct format!"
            )

        # Set a proper value for this, not too large or too small.
        self.n_iter = (
            n_iter if (n_iter is not None and n_iter > 0) else 2 * self.n_params
        )

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
            warnings.warn("Overwrite existing fit history!")
            self._history = []

        best_line_params_ = None
        for i in range(self.n_iter):
            param_id = i % self.n_params
            if best_line_params_ is None:
                last_params = [values[0] for name, values in self.param_grid]
            else:
                last_params = [
                    best_line_params_[name] for name, values in self.param_grid
                ]

            param_line = {}
            for pid, ((name, values), last_value) in enumerate(
                zip(self.param_grid, last_params)
            ):
                param_line[name] = [last_value] if pid != param_id else values

            grid_search = GridSearch(
                estimator=self.estimator,
                param_grid=param_line,
                opt_selection_method=self.opt_selection_methods[param_id],
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                refit=self.refit,
                cv=self.cv,
                verbose=self.verbose,
                pre_dispatch=self.pre_dispatch,
                error_score=self.error_score,
                return_train_score=self.return_train_score,
            )
            grid_search.fit(X=X, y=y, groups=groups, **fit_params)
            best_line_params_ = deepcopy(grid_search.best_params_)
            self._history.append(grid_search)

        # Buffer fitted attributes into LineSearch object.
        attrs = [
            v
            for v in vars(self._history[-1])
            if v.endswith("_") and not v.startswith("__")
        ]
        for attr in attrs:
            setattr(self, attr, getattr(self._history[-1], attr))
        return self

    def _run_search(self, evaluate_candidates):
        """Muted function."""
        return
