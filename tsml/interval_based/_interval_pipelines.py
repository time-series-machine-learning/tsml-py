"""Interval Extraction Pipeline Estimators.

Pipeline estimators using summary statistics extracted from random or supervised
 intervals and an estimator.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "RandomIntervalClassifier",
    "RandomIntervalRegressor",
    "SupervisedIntervalClassifier",
]

from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._base import _set_random_states
from sklearn.utils.validation import check_is_fitted, check_random_state

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations._interval_extraction import (
    RandomIntervalTransformer,
    SupervisedIntervalTransformer,
)
from tsml.utils.validation import check_n_jobs


class RandomIntervalClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Random Interval Classifier.

    Extracts multiple intervals with random length, position and dimension from series
    qnd concatenates them into a feature vector. Builds an estimator on the
    transformed data.

    Parameters
    ----------
    n_intervals : int or callable, default=100,
        The number of intervals of random length, position and dimension to be
        extracted.  Input should be an int or a function that takes a 3D np.ndarray
        input and returns an int.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    max_interval_length : int, default=3
        The maximum length of extracted intervals. Minimum value of min_interval_length.
    features : TransformerMixin, a function taking a 2d numpy array parameter, or list
            of said transformers and functions, default=None
        Transformers and functions used to extract features from selected intervals.
        If None, defaults to [mean, median, min, max, std, 25% quantile, 75% quantile]
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals and
        shapelets. If None, use the series as is.

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    dilation : int, list or None, default=None
        Add dilation to extracted intervals. No dilation is added if None or 1. If a
        list of ints, a random dilation value is selected from the list for each
        interval.

        The dilation value is selected after the interval star and end points. If the
        number of values in the dilated interval is less than the min_interval_length,
        the amount of dilation applied is reduced.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    random_state : None, int or instance of RandomState, default=None
        Seed or RandomState object used for random number generation.
        If random_state is None, use the RandomState singleton used by np.random.
        If random_state is an int, use a new RandomState instance seeded with seed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform` functions.
        `-1` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    class_dictionary_ : dict
        A dictionary mapping class labels to class indices in classes_.

    See Also
    --------
    RandomIntervalTransformer
    RandomIntervalRegressor
    SupervisedIntervalClassifier

    Examples
    --------
    >>> from tsml.interval_based import RandomIntervalClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> clf = RandomIntervalClassifier(random_state=0)
    >>> clf.fit(X, y)
    RandomIntervalClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 1, 0, 0, 1, 0, 1])
    """

    def __init__(
        self,
        n_intervals=100,
        min_interval_length=3,
        max_interval_length=np.inf,
        features=None,
        series_transformers=None,
        dilation=None,
        estimator=None,
        n_jobs=1,
        random_state=None,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.features = features
        self.series_transformers = series_transformers
        self.dilation = dilation
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super(RandomIntervalClassifier, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=3
        )
        X = self._convert_X(X)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)

        if isinstance(self.series_transformers, (list, tuple)):
            self._series_transformers = [
                None if st is None else _clone_estimator(st, random_state=rng)
                for st in self.series_transformers
            ]
        else:
            self._series_transformers = [
                None
                if self.series_transformers is None
                else _clone_estimator(self.series_transformers, random_state=rng)
            ]

        X_t = np.empty((X.shape[0], 0))
        self._transformers = []
        for st in self._series_transformers:
            if st is not None:
                s = st.fit_transform(X, y)
            else:
                s = X

            ct = RandomIntervalTransformer(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                max_interval_length=self.max_interval_length,
                features=self.features,
                dilation=self.dilation,
                n_jobs=self._n_jobs,
                parallel_backend=self.parallel_backend,
            )
            _set_random_states(ct, rng)
            self._transformers.append(ct)
            t = ct.fit_transform(s, y)

            X_t = np.hstack((X_t, t))

        self._estimator = _clone_estimator(
            RandomForestClassifier(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        self._estimator.fit(X_t, y)

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)
        X = self._convert_X(X)

        X_t = np.empty((X.shape[0], 0))
        for i, st in enumerate(self._series_transformers):
            if st is not None:
                s = st.transform(X)
            else:
                s = X

            t = self._transformers[i].transform(s)
            X_t = np.hstack((X_t, t))

        return self._estimator.predict(X_t)

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)
        X = self._convert_X(X)

        X_t = np.empty((X.shape[0], 0))
        for i, st in enumerate(self._series_transformers):
            if st is not None:
                s = st.transform(X)
            else:
                s = X

            t = self._transformers[i].transform(s)
            X_t = np.hstack((X_t, t))

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, self.class_dictionary_[preds[i]]] = 1
            return dists

    @classmethod
    def get_test_params(
        cls, parameter_set: Union[str, None] = None
    ) -> Union[dict, List[dict]]:
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from tsml.utils.numba_functions.stats import row_mean, row_numba_min

        return {
            "n_intervals": 2,
            "estimator": RandomForestClassifier(n_estimators=2),
            "features": [row_mean, row_numba_min],
        }


class RandomIntervalRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Random Interval Regressor.

    Extracts multiple intervals with random length, position and dimension from series
    and concatenates them into a feature vector. Builds an estimator on the
    transformed data.

    Parameters
    ----------
    n_intervals : int or callable, default=100,
        The number of intervals of random length, position and dimension to be
        extracted.  Input should be an int or a function that takes a 3D np.ndarray
        input and returns an int.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    max_interval_length : int, default=3
        The maximum length of extracted intervals. Minimum value of min_interval_length.
    features : TransformerMixin, a function taking a 2d numpy array parameter, or list
            of said transformers and functions, default=None
        Transformers and functions used to extract features from selected intervals.
        If None, defaults to [mean, median, min, max, std, 25% quantile, 75% quantile]
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals and
        shapelets. If None, use the series as is.

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    dilation : int, list or None, default=None
        Add dilation to extracted intervals. No dilation is added if None or 1. If a
        list of ints, a random dilation value is selected from the list for each
        interval.

        The dilation value is selected after the interval star and end points. If the
        number of values in the dilated interval is less than the min_interval_length,
        the amount of dilation applied is reduced.
    estimator : sklearn regressor, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestRegressor(n_estimators=200)
    random_state : None, int or instance of RandomState, default=None
        Seed or RandomState object used for random number generation.
        If random_state is None, use the RandomState singleton used by np.random.
        If random_state is an int, use a new RandomState instance seeded with seed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform` functions.
        `-1` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.

    See Also
    --------
    RandomIntervalTransformer
    RandomIntervalClassifier

    Examples
    --------
    >>> from tsml.interval_based import RandomIntervalRegressor
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10,
    ...                              regression_target=True, random_state=0)
    >>> reg = RandomIntervalRegressor(random_state=0)
    >>> reg.fit(X, y)
    RandomIntervalRegressor(...)
    >>> reg.predict(X)
    array([0.44924979, 1.31424037, 1.11951504, 0.63780969, 0.58123516,
           1.17135463, 0.56450198, 1.10128837])
    """

    def __init__(
        self,
        n_intervals=100,
        min_interval_length=3,
        max_interval_length=np.inf,
        features=None,
        series_transformers=None,
        dilation=None,
        estimator=None,
        n_jobs=1,
        random_state=None,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.features = features
        self.series_transformers = series_transformers
        self.dilation = dilation
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super(RandomIntervalRegressor, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The target labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=3
        )
        X = self._convert_X(X)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)
        rng = check_random_state(self.random_state)

        if isinstance(self.series_transformers, (list, tuple)):
            self._series_transformers = [
                None if st is None else _clone_estimator(st, random_state=rng)
                for st in self.series_transformers
            ]
        else:
            self._series_transformers = [
                None
                if self.series_transformers is None
                else _clone_estimator(self.series_transformers, random_state=rng)
            ]

        X_t = np.empty((X.shape[0], 0))
        self._transformers = []
        for st in self._series_transformers:
            if st is not None:
                s = st.fit_transform(X, y)
            else:
                s = X

            ct = RandomIntervalTransformer(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                max_interval_length=self.max_interval_length,
                features=self.features,
                dilation=self.dilation,
                n_jobs=self._n_jobs,
                parallel_backend=self.parallel_backend,
            )
            _set_random_states(ct, rng)
            self._transformers.append(ct)
            t = ct.fit_transform(s, y)

            X_t = np.hstack((X_t, t))

        self._estimator = _clone_estimator(
            RandomForestRegressor(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        self._estimator.fit(X_t, y)

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted target labels.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)
        X = self._convert_X(X)

        X_t = np.empty((X.shape[0], 0))
        for i, st in enumerate(self._series_transformers):
            if st is not None:
                s = st.transform(X)
            else:
                s = X

            t = self._transformers[i].transform(s)
            X_t = np.hstack((X_t, t))

        return self._estimator.predict(X_t)

    @classmethod
    def get_test_params(
        cls, parameter_set: Union[str, None] = None
    ) -> Union[dict, List[dict]]:
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from tsml.utils.numba_functions.stats import row_mean, row_numba_min

        return {
            "n_intervals": 3,
            "estimator": RandomForestRegressor(n_estimators=2),
            "features": [row_mean, row_numba_min],
        }


class SupervisedIntervalClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Supervised Interval Classifier.

    Extracts multiple intervals from series with using a supervised process
    and concatenates them into a feature vector. Builds an estimator on the
    transformed data.

    Parameters
    ----------
    n_intervals : int, default=50
        The number of times the supervised interval selection process is run. This
        process will extract more then one interval per run.
        Each supervised extraction will output a varying amount of features based on
        series length, number of dimensions and the number of features.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    features : callable, list of callables, default=None
        Functions used to extract features from selected intervals. Must take a 2d
        array of shape (n_instances, interval_length) and return a 1d array of shape
        (n_instances) containing the features.
        If None, defaults to the following statistics used in [2]:
        [mean, median, std, slope, min, max, iqr, count_mean_crossing,
        count_above_mean].
    metric : ["fisher"] or callable, default="fisher"
        The metric used to evaluate the usefulness of a feature extracted on an
        interval. If "fisher", the Fisher score is used. If a callable, it must take
        a 1d array of shape (n_instances) and return a 1d array of scores of shape
        (n_instances).
    randomised_split_point : bool, default=True
        If True, the split point for interval extraction is randomised as is done in [2]
        rather than split in half.
    normalise_for_search : bool, default=True
        If True, the data is normalised for the supervised interval search process.
        Features extracted for the transform output will not use normalised data.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    random_state : None, int or instance of RandomState, default=None
        Seed or RandomState object used for random number generation.
        If random_state is None, use the RandomState singleton used by np.random.
        If random_state is an int, use a new RandomState instance seeded with seed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform` functions.
        `-1` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    class_dictionary_ : dict
        A dictionary mapping class labels to class indices in classes_.

    See Also
    --------
    SupervisedIntervalTransformer
    RandomIntervalClassifier

    Examples
    --------
    >>> from tsml.interval_based import SupervisedIntervalClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> clf = SupervisedIntervalClassifier(random_state=0)
    >>> clf.fit(X, y)
    SupervisedIntervalClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 1, 0, 0, 1, 0, 1])
    """

    def __init__(
        self,
        n_intervals=50,
        min_interval_length=3,
        features=None,
        metric="fisher",
        randomised_split_point=True,
        normalise_for_search=True,
        estimator=None,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.features = features
        self.metric = metric
        self.randomised_split_point = randomised_split_point
        self.normalise_for_search = normalise_for_search
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super(SupervisedIntervalClassifier, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=7
        )
        X = self._convert_X(X)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = SupervisedIntervalTransformer(
            n_intervals=self.n_intervals,
            min_interval_length=self.min_interval_length,
            features=self.features,
            metric=self.metric,
            randomised_split_point=self.randomised_split_point,
            normalise_for_search=self.normalise_for_search,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            parallel_backend=self.parallel_backend,
        )

        self._estimator = _clone_estimator(
            RandomForestClassifier(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        self._estimator.fit(X_t, y)

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=7)
        X = self._convert_X(X)

        return self._estimator.predict(self._transformer.transform(X))

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=7)
        X = self._convert_X(X)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self.class_dictionary_[preds[i]]] = 1
            return dists

    @classmethod
    def get_test_params(
        cls, parameter_set: Union[str, None] = None
    ) -> Union[dict, List[dict]]:
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from tsml.utils.numba_functions.stats import row_mean, row_numba_min

        return {
            "n_intervals": 1,
            "estimator": RandomForestClassifier(n_estimators=2),
            "features": [row_mean, row_numba_min],
        }
