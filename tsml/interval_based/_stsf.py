"""Supervised Time Series Forest classifiers."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["STSFClassifier", "RSTSFClassifier"]

from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.interval_based._base import BaseIntervalForest
from tsml.transformations import (
    ARCoefficientTransformer,
    FunctionTransformer,
    PeriodogramTransformer,
    SupervisedIntervalTransformer,
)
from tsml.utils.numba_functions.general import first_order_differences_3d
from tsml.utils.numba_functions.stats import (
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_slope,
    row_std,
)
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class STSFClassifier(ClassifierMixin, BaseIntervalForest):
    """Supervised Time Series Forest (STSF).

    An ensemble of decision trees built on intervals selected through a supervised
    process as described in _[1].
    Overview: Input n series length m
    For each tree
        - sample X using class-balanced bagging
        - sample intervals for all 3 representations and 7 features using supervised
        - method
        - find mean, median, std, slope, iqr, min and max using their corresponding
        - interval for each representation, concatenate to form new data set
        - build a decision tree on new data set
    Ensemble the trees with averaged probability estimates.

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    min_interval_length : int, float, list, or tuple, default=3
        Minimum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the minimum interval length.

        Different minimum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    use_pyfftw : bool, default=True
        Whether to use the pyfftw library for FFT calculations. Requires the pyfftw
        package to be installed.
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
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of TransformerMixin
        Stores the interval extraction transformer for all estimators.
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances_ ,total_intervals * att_subsample_size)
        The transformed dataset for all estimators. Only saved when
        save_transformed_data is true.

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/STSF.java>`_.

    References
    ----------
    .. [1] Cabello, Nestor, et al. "Fast and Accurate Time Series Classification
       Through Supervised Interval Search." IEEE ICDM 2020

    Examples
    --------
    >>> from tsml.interval_based import STSFClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=10, series_length=12, random_state=0)
    >>> clf = STSFClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    STSFClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        min_interval_length=3,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pyfftw=True,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.use_pyfftw = use_pyfftw
        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)

        series_transformers = [
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(use_pyfftw=use_pyfftw),
        ]

        interval_features = [
            row_mean,
            row_std,
            row_slope,
            row_median,
            row_iqr,
            row_numba_min,
            row_numba_max,
        ]

        super(STSFClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="supervised",
            n_intervals=1,
            min_interval_length=min_interval_length,
            max_interval_length=np.inf,
            interval_features=interval_features,
            series_transformers=series_transformers,
            att_subsample_size=None,
            replace_nan=0,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

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
        return self._predict_proba(X)

    def _more_tags(self) -> dict:
        return {
            "optional_dependency": self.use_pyfftw,
        }

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
        return {
            "n_estimators": 2,
        }


class RSTSFClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Random Supervised Time Series Forest (RSTSF) Classifier.

    An ensemble of decision trees built on intervals selected through a supervised
    process as described in _[1].
    Overview: Input n series of length m with d dimensions
        - sample X using class-balanced bagging
        - sample intervals for all 4 series representations and 9 features using
            supervised method
        - build extra trees classifier on transformed interval data

    Parameters
    ----------
    n_estimators : int, default=200
        The number of trees in the forest.
    n_intervals : int, default=50
        The number of times the supervised interval selection process is run.
        Each supervised extraction will output a varying amount of features based on
        series length, number of dimensions and the number of features.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    use_pyfftw : bool, default=True
        Whether to use pyfftw for the periodogram transformation.
    random_state : None, int or instance of RandomState, default=None
        Seed or RandomState object used for random number generation.
        If random_state is None, use the RandomState singleton used by np.random.
        If random_state is an int, use a new RandomState instance seeded with seed.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict` functions.
        `-1` means using all processors.

    See Also
    --------
    SupervisedIntervals

    References
    ----------
    .. [1] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2021. Fast, accurate and
        interpretable time series classification through randomization. arXiv preprint
        arXiv:2105.14876.

    Examples
    --------
    >>> from tsml.interval_based import RSTSFClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=10, series_length=12, random_state=0)
    >>> clf = RSTSFClassifier(n_estimators=10, n_intervals=5, random_state=0)
    >>> clf.fit(X, y)
    RSTSFClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    def __init__(
        self,
        n_estimators=200,
        n_intervals=50,
        min_interval_length=3,
        use_pyfftw=True,
        random_state=None,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.use_pyfftw = use_pyfftw
        self.random_state = random_state
        self.n_jobs = n_jobs

        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)
        _check_optional_dependency("statsmodels", "statsmodels", self)

        super(RSTSFClassifier, self).__init__()

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
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=5
        )
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        lags = int(12 * (X.shape[2] / 100.0) ** 0.25)

        self._series_transformers = [
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(use_pyfftw=self.use_pyfftw),
            ARCoefficientTransformer(order=lags, replace_nan=True),
        ]

        transforms = [X] + [t.fit_transform(X) for t in self._series_transformers]

        Xt = np.empty((X.shape[0], 0))
        self._transformers = []
        transform_data_lengths = []
        for t in transforms:
            si = SupervisedIntervalTransformer(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
                randomised_split_point=True,
            )
            features = si.fit_transform(t, y)
            Xt = np.hstack((Xt, features))
            self._transformers.append(si)
            transform_data_lengths.append(features.shape[1])

        self.clf_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion="entropy",
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        self.clf_.fit(Xt, y)

        relevant_features = []
        for tree in self.clf_.estimators_:
            relevant_features.extend(tree.tree_.feature[tree.tree_.feature >= 0])
        relevant_features = np.unique(relevant_features)

        features_to_transform = [False] * Xt.shape[1]
        for i in relevant_features:
            features_to_transform[i] = True

        count = 0
        for r in range(len(transforms)):
            self._transformers[r].set_features_to_transform(
                features_to_transform[count : count + transform_data_lengths[r]],
                raise_error=False,
            )
            count += transform_data_lengths[r]

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

        Xt = self._predict_transform(X)
        return self.clf_.predict(Xt)

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

        Xt = self._predict_transform(X)
        return self.clf_.predict_proba(Xt)

    def _predict_transform(self, X):
        X = self._validate_data(X=X, ensure_min_series_length=5, reset=False)
        X = self._convert_X(X)

        transforms = [X] + [t.transform(X) for t in self._series_transformers]

        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            si = self._transformers[i]
            Xt = np.hstack((Xt, si.transform(t)))

        return Xt

    def _more_tags(self) -> dict:
        return {
            "optional_dependency": True,
        }

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
        return {
            "n_estimators": 2,
            "n_intervals": 2,
        }
