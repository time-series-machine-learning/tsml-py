# -*- coding: utf-8 -*-
"""A base class for interval extracting forest estimators"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["BaseIntervalForest"]

import inspect
import time
import warnings

import numpy as np
from joblib import Parallel
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations.interval_extraction import (
    RandomIntervalTransformer,
    SupervisedIntervalTransformer,
)
from tsml.utils.numba_functions.stats import row_mean, row_slope, row_std
from tsml.utils.validation import check_n_jobs, is_transformer
from tsml.vector import CITClassifier


class BaseIntervalForest(BaseTimeSeriesEstimator):
    """A base class for interval extracting forest estimators.

    Allows the implementation of classifiers and regressors along the lines of [1][2][3]
    which extract intervals and create an ensemble from the subsequent features.

        Extension of the CIF algorithm using multple representations. Implementation of the
    interval based forest making use of the catch22 feature set on randomly selected
    intervals on the base series, periodogram representation and differences
    representation described in the HIVE-COTE 2.0 paper Middlehurst et al (2021). [1]_

    Overview: Input "n" series with "d" dimensions of length "m".
    For each tree
        - Sample n_intervals intervals per representation of random position and length
        - Subsample att_subsample_size catch22 or summary statistic attributes randomly
        - Randomly select dimension for each interval
        - Calculate attributes for each interval from its representation, concatenate
          to form new data set
        - Build decision tree on new data set
    Ensemble the trees with averaged probability estimates

    Parameters
    ----------
    base_estimator : BaseEstimator
        Base estimator for the ensemble, can be supplied a sklearn BaseEstimator or a
        string for suggested options.
    estimator_type : str

        self.interval_selection_method = interval_selection_method
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.interval_features = interval_features
        self.series_transformers = series_transformers
        self.att_subsample_size = att_subsample_size
        self.replace_nan = replace_nan

    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    interval_selection_method :

    n_intervals : int, length 3 list of int or None, default=None
        Number of intervals to extract per representation per tree as an int for all
        representations or list for individual settings, if None extracts
        (4 + (sqrt(representation_length) * sqrt(n_dims)) / 3) intervals.
    min_interval_length : int or length 3 list of int, default=4
        Minimum length of an interval per representation as an int for all
        representations or list for individual settings.
    max_interval_length : int, length 3 list of int or None, default=None
        Maximum length of an interval per representation as an int for all
        representations or list for individual settings, if None set to
        (representation_length / 2).
    interval_features :

    series_transformers :

    att_subsample_size : int, default=10
        Number of catch22 or summary statistic attributes to subsample per tree.
    replace_nan :

    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of ndarray with shape (total_intervals,2)
        Stores indexes of each intervals start and end points for all classifiers.
    atts_ : list of shape (n_estimators) of array with shape (att_subsample_size)
        Attribute indexes of the subsampled catch22 or summary statistic for all
        classifiers.
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances,total_intervals * att_subsample_size)
        The transformed dataset for all classifiers. Only saved when
        save_transformed_data is true.

    References
    ----------
    .. [1]
    .. [2]
    .. [3]
    """

    def __init__(
        self,
        base_estimator,
        n_estimators=200,
        interval_selection_method="random",
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        interval_features=None,
        series_transformers=None,
        att_subsample_size=None,
        replace_nan=None,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.interval_selection_method = interval_selection_method
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.interval_features = interval_features
        self.series_transformers = series_transformers
        self.att_subsample_size = att_subsample_size
        self.replace_nan = replace_nan
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    # if subsampling attributes, an interval_features transformer must contain a
    # parameter name from transformer_feature_selection and an attribute name from
    # transformer_feature_names to allow features to be subsampled
    transformer_feature_selection = ["features"]
    transformer_feature_names = ["features_arguments_"]
    # an interval_features transformer must contain one of these attribute names to
    # be able to skip transforming features in predict
    transformer_feature_skip = ["transform_features_", "_transform_features"]

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
    }

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        if is_classifier(self):
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            self.class_dictionary_ = {}
            for index, classVal in enumerate(self.classes_):
                self.class_dictionary_[classVal] = index

        # default base_estimators for classification and regression
        self._base_estimator = self.base_estimator
        if self.base_estimator is None:
            if is_classifier(self):
                self._base_estimator = DecisionTreeClassifier(criterion="entropy")
            elif is_regressor(self):
                self._base_estimator = DecisionTreeRegressor(criterion="absolute_error")
            else:
                raise ValueError()  # todo error for invalid self.base_estimator
        # base_estimator must be an sklearn estimator
        elif not isinstance(self.base_estimator, BaseEstimator):
            raise ValueError()  # todo error for invalid self.base_estimator

        # use the base series if series_transformers is None
        if self.series_transformers is None or self.series_transformers == []:
            Xt = [X]
            self._series_transformers = [None]
        # clone series_transformers if it is a transformer and transform the input data
        elif is_transformer(self.series_transformers):
            t = _clone_estimator(
                self.series_transformers, random_state=self.random_state
            )
            Xt = [t.fit_transform(X, y)]
            self._series_transformers = [t]
        # clone each series_transformers transformer and include the base series if None
        # is in the list
        elif isinstance(self.series_transformers, (list, tuple)):
            Xt = []
            self._series_transformers = []

            for transformer in self.series_transformers:
                if transformer is None:
                    Xt.append(X)
                    self._series_transformers.append(None)
                elif is_transformer(transformer):
                    t = _clone_estimator(transformer, random_state=self.random_state)
                    Xt.append(t.fit_transform(X, y))
                    self._series_transformers.append(t)
                else:
                    raise ValueError()  # todo error for invalid self.series_transformers
        # other inputs are invalid
        else:
            raise ValueError()  # todo error for invalid self.series_transformers

        # if only a single n_intervals value is passed it must be an int or str
        if isinstance(self.n_intervals, (int, str)):
            n_intervals = [[self.n_intervals]] * len(Xt)
        elif isinstance(self.n_intervals, (list, tuple)):
            # if only one series_transformer is used, n_intervals can be a list of
            # multiple n_intervals options to be applied
            if len(Xt) == 1:
                for method in self.n_intervals:
                    if not isinstance(method, (int, str)):
                        raise ValueError()  # todo error for invalid self.n_intervals
                n_intervals = [self.n_intervals]
            # if more than one series_transformer is used, n_intervals must have the
            # same number of items if it is a list
            elif len(self.n_intervals) != len(Xt):
                raise ValueError()  # todo error for invalid self.n_intervals
            # list items can be a list of items or a single item for each
            # series_transformer, but each individual item must be an int or str
            else:
                n_intervals = []
                for features in self.n_intervals:
                    if isinstance(features, (list, tuple)):
                        for method in features:
                            if not isinstance(method, (int, str)):
                                raise ValueError()  # todo error for invalid self.n_intervals
                        n_intervals.append(features)
                    elif isinstance(features, (int, str)):
                        n_intervals.append([features])
                    else:
                        raise ValueError()  # todo error for invalid self.n_intervals
        # other inputs are invalid
        else:
            raise ValueError()  # todo error for invalid self.n_intervals

        # add together the number of intervals for each series_transformer
        # str input must be one of a set valid options
        self._n_intervals = [0] * len(Xt)
        for i, series in enumerate(Xt):
            for method in n_intervals[i]:
                if isinstance(method, int):
                    self._n_intervals[i] += method
                elif isinstance(method, str):
                    # sqrt of series length
                    if method.lower() == "sqrt":
                        self._n_intervals[i] += int(
                            np.sqrt(series.shape[2]) * np.sqrt(series.shape[1])
                        )
                    # sqrt of series length divided by the number of series_transformers
                    elif method.lower() == "sqrt-div":
                        self._n_intervals[i] += int(
                            (np.sqrt(series.shape[2]) * np.sqrt(series.shape[1]))
                            / len(Xt)
                        )
                    else:
                        raise ValueError()  # todo error for invalid self.n_intervals string

        # each series_transformer must have at least 1 interval extracted
        for i, n in enumerate(self._n_intervals):
            if n <= 0:
                self._n_intervals[i] = 1

        self.total_intervals_ = sum(self._n_intervals)

        # minimum interval length
        if isinstance(self.min_interval_length, int):
            self._min_interval_length = [self.min_interval_length] * len(Xt)
        # min_interval_length must be at less than one if it is a float (proportion of
        # total attributed to subsample)
        elif (
            isinstance(self.min_interval_length, float)
            and self.min_interval_length <= 1
        ):
            self._min_interval_length = [
                int(self.min_interval_length * t.shape[2]) for t in Xt
            ]
        # if the input is a list, it must be the same length as the number of
        # series_transformers
        # list values must be ints or floats. The same checks as above are performed
        elif isinstance(self.min_interval_length, (list, tuple)):
            if len(self.min_interval_length) == len(Xt):
                raise ValueError()  # todo error for invalid self.min_interval_length string

            self._min_interval_length = []
            for i, length in enumerate(self.min_interval_length):
                if isinstance(length, float) and length <= 1:
                    self._min_interval_length.append(int(length * Xt[i].shape[2]))
                elif isinstance(length, int):
                    self._min_interval_length.append(length)
                else:
                    raise ValueError()  # todo error for invalid self.min_interval_length string
        # other inputs are invalid
        else:
            raise ValueError()  # todo error for invalid self.min_interval_length string

        # min_interval_length cannot be less than 3 or greater than the series length
        for i, n in enumerate(self._min_interval_length):
            if n > Xt[i].shape[2]:
                self._min_interval_length[i] = Xt[i].shape[2]
            elif n < 3:
                self._min_interval_length[i] = 3

        # maximum interval length
        if (
            isinstance(self.max_interval_length, int)
            or self.max_interval_length == np.inf
        ):
            self._max_interval_length = [self.max_interval_length] * len(Xt)
        # max_interval_length must be at less than one if it is a float (proportion of
        # total attributed to subsample)
        elif (
            isinstance(self.max_interval_length, float)
            and self.max_interval_length <= 1
        ):
            self._max_interval_length = [
                int(self.max_interval_length * t.shape[2]) for t in Xt
            ]
        # if the input is a list, it must be the same length as the number of
        # series_transformers
        # list values must be ints or floats. The same checks as above are performed
        elif isinstance(self.max_interval_length, (list, tuple)):
            if len(self.max_interval_length) == len(Xt):
                raise ValueError()  # todo error for invalid self.max_interval_length string

            self._max_interval_length = []
            for i, length in enumerate(self.max_interval_length):
                if isinstance(length, float) and length <= 1:
                    self._max_interval_length.append(int(length * Xt[i].shape[2]))
                elif isinstance(length, int):
                    self._max_interval_length.append(length)
                else:
                    raise ValueError()  # todo error for invalid self.max_interval_length string
        # other inputs are invalid
        else:
            raise ValueError()  # todo error for invalid self.max_interval_length string

        # max_interval_length cannot be less than min_interval_length or greater than
        # the series length
        for i, n in enumerate(self._max_interval_length):
            if n < self._min_interval_length[i]:
                self._max_interval_length[i] = self._min_interval_length[i]
            elif n > Xt[i].shape[2]:
                self._max_interval_length[i] = Xt[i].shape[2]

        # we store whether each series_transformer contains a transformer and/or
        # function in its interval_features
        self._interval_transformer = [False] * len(Xt)
        self._interval_function = [False] * len(Xt)
        # single transformer or function for all series_transformers
        if is_transformer(self.interval_features):
            self._interval_transformer = [True] * len(Xt)
            self._interval_features = [[self.interval_features]] * len(Xt)
        elif callable(self.interval_features):
            self._interval_function = [True] * len(Xt)
            self._interval_features = [[self.interval_features]] * len(Xt)
        elif isinstance(self.interval_features, (list, tuple)):
            # if only one series_transformer is used, n_intervals can be a list of
            # multiple n_intervals options to be applied todo
            if len(Xt) == 1:
                for i, feature in enumerate(self.interval_features):
                    if is_transformer(feature):
                        self._interval_transformer[0] = True
                    elif callable(feature):
                        self._interval_function[0] = True
                    else:
                        raise ValueError()  # todo error for invalid self.interval_features
                self._interval_features = [self.interval_features]
            # if more than one series_transformer is used, n_intervals must have the
            # same number of items if it is a list todo
            elif len(self.interval_features) != len(Xt):
                raise ValueError()  # todo error for invalid self.interval_features
            # list items can be a list of items or a single item for each
            # series_transformer, but each individual item must be an int or str todo
            else:
                self._interval_features = []
                for i, feature in enumerate(self.interval_features):
                    if isinstance(feature, (list, tuple)):
                        for method in feature:
                            if is_transformer(method):
                                self._interval_transformer[i] = True
                            elif callable(method):
                                self._interval_function[i] = True
                            else:
                                raise ValueError()  # todo error for invalid self.interval_features
                        self._interval_features.append(feature)
                    elif is_transformer(feature):
                        self._interval_transformer[i] = True
                        self._interval_features.append([feature])
                    elif callable(feature):
                        self._interval_function[i] = True
                        self._interval_features.append([feature])
                    else:
                        raise ValueError()  # todo error for invalid self.interval_features
        # use basic summary stats by default if None
        elif self.interval_features is None:
            self._interval_function = [True] * len(Xt)
            self._interval_features = [[row_mean, row_std, row_slope]] * len(Xt)
        # other inputs are invalid
        else:
            raise ValueError()  # todo error for invalid self.interval_features

        # att_subsample_size must be at least one if it is an int
        if isinstance(self.att_subsample_size, int):
            if self.att_subsample_size < 1:
                raise ValueError()  # todo error for invalid invalid self.att_subsample_size

            self._att_subsample_size = [self.att_subsample_size] * len(Xt)
        # att_subsample_size must be at less than one if it is a float (proportion of
        # total attributed to subsample)
        elif isinstance(self.att_subsample_size, float):
            if self.att_subsample_size > 1:
                raise ValueError()  # todo error for invalid invalid self.att_subsample_size

            self._att_subsample_size = [self.att_subsample_size] * len(Xt)
        # default is no attribute subsampling with None
        elif self.att_subsample_size is None:
            self._att_subsample_size = [self.att_subsample_size] * len(Xt)
        # if the input is a list, it must be the same length as the number of
        # series_transformers
        # list values must be ints, floats or None. The same checks as above are
        # performed
        elif isinstance(self.att_subsample_size, (list, tuple)):
            if len(self.att_subsample_size) != len(Xt):
                raise ValueError()  # todo error for invalid self.att_subsample_size

            self._att_subsample_size = []
            for ssize in self.att_subsample_size:
                if isinstance(ssize, int):
                    if ssize < 1:
                        raise ValueError()  # todo error for invalid invalid self.att_subsample_size

                    self._att_subsample_size.append(ssize)
                elif isinstance(ssize, float):
                    if ssize > 1:
                        raise ValueError()  # todo error for invalid invalid self.att_subsample_size

                    self._att_subsample_size.append(ssize)
                elif ssize is None:
                    self._att_subsample_size.append(ssize)
                else:
                    raise ValueError()  # todo error for invalid self.att_subsample_size
        # other inputs are invalid
        else:
            raise ValueError()  # todo error for invalid invalid self.att_subsample_size

        # if we are subsampling attributes for a series_transformer and it uses a
        # BaseTransformer, we must ensure it has the required parameters and
        # attributes to do so
        self._transformer_feature_selection = [[]] * len(Xt)
        self._transformer_feature_names = [[]] * len(Xt)
        for r, att_subsample in enumerate(self._att_subsample_size):
            if att_subsample is not None:
                for transformer in self._interval_features[r]:
                    if is_transformer(transformer):
                        params = inspect.signature(transformer.__init__).parameters

                        # the BaseTransformer must have a parameter with one of the
                        # names listed in transformer_feature_selection as a way to
                        # select which features the transformer should transform
                        has_params = False
                        for n in self.transformer_feature_selection:
                            if params.get(n, None) is not None:
                                has_params = True
                                self._transformer_feature_selection[r].append(n)
                                break

                        if not has_params:
                            raise ValueError()  # todo error for invalid invalid self.att_subsample_size

                        # the BaseTransformer must have an attribute with one of the
                        # names listed in transformer_feature_names as a list or tuple
                        # of valid options for the previous parameter
                        has_feature_names = False
                        for n in self.transformer_feature_names:
                            if hasattr(transformer, n) and isinstance(
                                getattr(transformer, n), (list, tuple)
                            ):
                                has_feature_names = True
                                self._transformer_feature_names[r].append(n)
                                break

                        if not has_feature_names:
                            raise ValueError()  # todo error for invalid invalid self.att_subsample_size

        # verify the interval_selection_method is a valid string
        if isinstance(self.interval_selection_method, str):
            # SupervisedIntervals cannot currently handle transformers or regression
            if (
                self.interval_selection_method.lower() == "supervised"
                or self.interval_selection_method.lower() == "random-supervised"
            ):
                if any(self._interval_transformer):
                    raise ValueError()  # todo error for invalid invalid self.interval_selection_method

                if is_regressor(self):
                    raise ValueError()  # todo error for invalid invalid self.interval_selection_method
            # RandomIntervals
            elif not self.interval_selection_method.lower() == "random":
                raise ValueError()  # todo error for invalid invalid self.interval_selection_method
        # other inputs are invalid
        else:
            raise ValueError()  # todo error for invalid self.interval_selection_method

        # todo int option?
        # option to replace NaN values must be a valid string
        if isinstance(self.replace_nan, str):
            if (
                not self.replace_nan.lower() == "zero"
                and not self.replace_nan.lower() == "nan"
            ):
                raise ValueError()  # todo error for invalid self.replace_nan
        # other inputs are invalid except for None
        elif self.replace_nan is not None:
            raise ValueError()  # todo error for invalid self.replace_nan

        self._n_jobs = check_n_jobs(self.n_jobs)
        self._efficient_predictions = True  # todo
        self._test_flag = False  # todo

        self._n_estimators = self.n_estimators
        if self.time_limit_in_minutes is not None and self.time_limit_in_minutes > 0:
            time_limit = self.time_limit_in_minutes * 60
            start_time = time.time()
            train_time = 0

            self._n_estimators = 0
            self.estimators_ = []
            self.intervals_ = []
            self.transformed_data_ = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(
                    n_jobs=self._n_jobs,
                    backend=self.parallel_backend,
                    prefer="threads",
                )(
                    delayed(self._fit_estimator)(
                        Xt,
                        y,
                        i,
                    )
                    for i in range(self._n_jobs)
                )

                (
                    estimators,
                    intervals,
                    transformed_data,
                ) = zip(*fit)

                self.estimators_ += estimators
                self.intervals_ += intervals
                self.transformed_data_ += transformed_data

                self._n_estimators += self._n_jobs
                train_time = time.time() - start_time
        else:
            fit = Parallel(
                n_jobs=self._n_jobs,
                backend=self.parallel_backend,
                prefer="threads",
            )(
                delayed(self._fit_estimator)(
                    Xt,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            (
                self.estimators_,
                self.intervals_,
                self.transformed_data_,
            ) = zip(*fit)

        return self

    def predict(self, X):
        if is_regressor(self):
            Xt = self._predict_setup(X)

            y_preds = Parallel(
                n_jobs=self._n_jobs,
                backend=self.parallel_backend,
                prefer="threads",
            )(
                delayed(self._predict_for_estimator)(
                    Xt,
                    self.estimators_[i],
                    self.intervals_[i],
                    predict_proba=False,
                )
                for i in range(self._n_estimators)
            )

            return np.mean(y_preds, axis=0)
        else:
            return np.array(
                [self.classes_[int(np.argmax(prob))] for prob in self._predict_proba(X)]
            )

    def _predict_proba(self, X):
        Xt = self._predict_setup(X)

        y_probas = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._predict_for_estimator)(
                Xt,
                self.estimators_[i],
                self.intervals_[i],
                predict_proba=True,
            )
            for i in range(self._n_estimators)
        )

        output = np.sum(y_probas, axis=0) / (
            np.ones(self.n_classes_) * self._n_estimators
        )
        return output

    def _fit_estimator(self, Xt, y, i):
        # random state for this estimator
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (i + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        intervals = []
        transform_data_lengths = []
        interval_features = np.empty((self.n_instances_, 0))

        # for each transformed series
        for r in range(len(Xt)):
            # subsample attributes if enabled
            if self._att_subsample_size[r] is not None:
                # separate transformers and functions in separate lists
                # add the feature names of transformers to a list to subsample from
                # and calculate the total number of features
                all_transformers = []
                all_transformer_features = []
                all_function_features = []
                for feature in self._interval_features[r]:
                    if is_transformer(feature):
                        all_transformer_features += getattr(
                            feature,
                            self._transformer_feature_names[r][len(all_transformers)],
                        )
                        all_transformers.append(feature)
                    else:
                        all_function_features.append(feature)

                # handle float subsample size
                num_features = len(all_transformer_features) + len(
                    all_function_features
                )
                att_subsample_size = self._att_subsample_size[r]
                if isinstance(self._att_subsample_size[r], float):
                    att_subsample_size = int(att_subsample_size * num_features)

                # if the att_subsample_size is greater than the number of features
                # give a warning and add all features
                features = []
                if att_subsample_size < num_features:
                    # subsample the transformer and function features by index
                    atts = rng.choice(
                        num_features,
                        att_subsample_size,
                        replace=False,
                    )
                    atts.sort()

                    # subsample the feature transformers using the
                    # transformer_feature_names and transformer_feature_selection
                    # attributes.
                    # the presence of valid attributes is verified in fit.
                    count = 0
                    length = 0
                    for n, transformer in enumerate(all_transformers):
                        this_len = len(
                            getattr(transformer, self._transformer_feature_names[r][n])
                        )
                        length += this_len

                        # subsample feature names from this transformer
                        t_features = []
                        while count < len(atts) and atts[count] < length:
                            t_features.append(
                                getattr(
                                    transformer,
                                    self._transformer_feature_names[r][n],
                                )[atts[count] + this_len - length]
                            )
                            count += 1

                        # tell this transformer to only transform the selected features
                        if len(t_features) > 0:
                            new_transformer = _clone_estimator(transformer, rs)
                            setattr(
                                new_transformer,
                                self._transformer_feature_selection[r][n],
                                t_features,
                            )
                            features.append(new_transformer)

                    # subsample the remaining function features
                    for i in range(att_subsample_size - count):
                        features.append(all_function_features[atts[count + i] - length])
                else:
                    warnings.warn(
                        f"Attribute subsample size {att_subsample_size} is larger than or equal to the number of attributes {num_features} for series {self._series_transformers[r]}"
                    )
                    for feature in self._interval_features[r]:
                        if is_transformer(feature):
                            features.append(_clone_estimator(feature, rs))
                        else:
                            features.append(feature)
            # add all features while cloning estimators if not subsampling
            else:
                features = []
                for feature in self._interval_features[r]:
                    if is_transformer(feature):
                        features.append(_clone_estimator(feature, rs))
                    else:
                        features.append(feature)

            # create the selected interval selector and set its parameters
            if self.interval_selection_method == "random":
                selector = RandomIntervalTransformer(
                    n_intervals=self._n_intervals[r],
                    min_interval_length=self._min_interval_length[r],
                    max_interval_length=self._max_interval_length[r],
                    features=features,
                    random_state=rs,
                )
            elif self.interval_selection_method == "supervised":
                selector = SupervisedIntervalTransformer(
                    n_intervals=self._n_intervals[r],
                    min_interval_length=self._min_interval_length[r],
                    features=features,
                    randomised_split_point=False,
                    random_state=rs,
                )
            elif self.interval_selection_method == "random-supervised":
                selector = SupervisedIntervalTransformer(
                    n_intervals=self._n_intervals[r],
                    min_interval_length=self._min_interval_length[r],
                    features=features,
                    randomised_split_point=True,
                    random_state=rs,
                )
            else:
                raise ValueError()  # todo error for invalid self.interval_selection_method, should not get here

            # fit the interval selector, transform the current series using it and save
            # the transformer
            intervals.append(selector)
            f = intervals[r].fit_transform(Xt[r], y)

            # concatenate the data and save this transforms number of attributes
            transform_data_lengths.append(f.shape[1])
            interval_features = np.hstack((interval_features, f))

        # replace invalid attributes with 0 or np.nan if option is selected
        if self.replace_nan == "zero":
            interval_features = np.nan_to_num(interval_features, False, 0, 0, 0)
        elif self.replace_nan == "nan":
            interval_features = np.nan_to_num(
                interval_features, False, np.nan, np.nan, np.nan
            )

        # clone and fit the base estimator using the transformed data
        tree = _clone_estimator(self._base_estimator, random_state=rs)
        tree.fit(interval_features, y)

        # find the features used in the tree and inform the interval selectors to not
        # transform these features if possible
        if not self._test_flag:
            relevant_features = None
            if isinstance(tree, BaseDecisionTree):
                relevant_features = np.unique(
                    tree.tree_.feature[tree.tree_.feature >= 0]
                )
            elif isinstance(tree, CITClassifier):
                relevant_features, _ = tree.tree_node_splits_and_gain()

            if relevant_features is not None:
                features_to_transform = [False] * interval_features.shape[1]
                for i in relevant_features:
                    features_to_transform[i] = True

                count = 0
                for r in range(len(Xt)):
                    intervals[
                        r
                    ].transformer_feature_skip = self.transformer_feature_skip

                    # if the transformers don't have valid attributes to skip False is
                    # returned
                    completed = intervals[r].set_features_to_transform(
                        features_to_transform[
                            count : count + transform_data_lengths[r]
                        ],
                        raise_error=False,
                    )
                    count += transform_data_lengths[r]

                    if not completed:
                        self._efficient_predictions = False
            else:
                self._efficient_predictions = False
        else:
            self._efficient_predictions = False

        return [
            tree,
            intervals,
            interval_features if self.save_transformed_data else None,
        ]

    def _predict_setup(self, X):
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)

        n_instances, n_dims, series_length = X.shape

        if n_dims != self.n_dims_:
            raise ValueError(
                "The number of dimensions in the train data does not match the number "
                "of dimensions in the test data"
            )
        if series_length != self.series_length_:
            raise ValueError(
                "The series length of the train data does not match the series length "
                "of the test data"
            )

        Xt = []
        for transformer in self._series_transformers:
            if transformer is None:
                Xt.append(X)
            elif is_transformer(transformer):
                Xt.append(transformer.transform(X))

        return Xt

    def _predict_for_estimator(self, Xt, estimator, intervals, predict_proba=False):
        interval_features = np.empty((Xt[0].shape[0], 0))

        for r in range(len(Xt)):
            f = intervals[r].transform(Xt[r])
            interval_features = np.hstack((interval_features, f))

        if self.replace_nan == "zero":
            interval_features = np.nan_to_num(interval_features, False, 0, 0, 0)
        elif self.replace_nan == "nan":
            interval_features = np.nan_to_num(
                interval_features, False, np.nan, np.nan, np.nan
            )

        if predict_proba:
            return estimator.predict_proba(interval_features)
        else:
            return estimator.predict(interval_features)
