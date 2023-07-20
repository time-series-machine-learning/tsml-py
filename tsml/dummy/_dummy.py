"""Dummy time series estimators."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["DummyClassifier", "DummyRegressor", "DummyClusterer"]

from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _num_samples, check_is_fitted

from tsml.base import BaseTimeSeriesEstimator


class DummyClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more
    complex classifiers. Do not use it for real problems.

    The specific behavior of the baseline is selected with the `strategy`
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`. The predictions, however,
    typically depend on values observed in the `y` parameter passed to `fit`.

    A wrapper for `sklearn.dummy.DummyClassifier` using the tsml interface. Functionally
    identical.

    Parameters
    ----------
    strategy : {"most_frequent", "prior", "stratified", "uniform", \
            "constant"}, default="prior"
        Strategy to use to generate predictions.

        - "most_frequent": the `predict` method always returns the most
          frequent class label in the observed `y` argument passed to `fit`.
          The `predict_proba` method returns the matching one-hot encoded
          vector.
        - "prior": the `predict` method always returns the most frequent
          class label in the observed `y` argument passed to `fit` (like
          "most_frequent"). ``predict_proba`` always returns the empirical
          class distribution of `y` also known as the empirical class prior
          distribution.
        - "stratified": the `predict_proba` method randomly samples one-hot
          vectors from a multinomial distribution parametrized by the empirical
          class prior probabilities.
          The `predict` method returns the class label which got probability
          one in the one-hot vector of `predict_proba`.
          Each sampled row of both methods is therefore independent and
          identically distributed.
        - "uniform": generates predictions uniformly at random from the list
          of unique classes observed in `y`, i.e. each class has equal
          probability.
        - "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class.
    constant : int or str or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.
    validate : bool, default=False
        Whether to perform validation checks on X and y.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness to generate the predictions when
        ``strategy='stratified'`` or ``strategy='uniform'``.
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    See Also
    --------
    DummyRegressor : Regressor that makes predictions using simple rules.
    DummyClusterer : Clusterer that makes predictions using simple rules.

    Examples
    --------
    >>> from tsml.dummy import DummyClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> clf = DummyClassifier(strategy="most_frequent")
    >>> clf.fit(X, y)
    DummyClassifier(...)
    >>> clf.score(X, y)
    0.5
    """

    def __init__(
        self,
        strategy="prior",
        constant=None,
        validate=False,
        random_state=None,
    ):
        self.strategy = strategy
        self.constant = constant
        self.validate = validate
        self.random_state = random_state

        super(DummyClassifier, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        if self.validate:
            X, y = self._validate_data(X=X, y=y, ensure_min_series_length=1)

            check_classification_targets(y)

        self.classes_ = np.unique(np.asarray(y))

        if self.validate:
            self.n_classes_ = self.classes_.shape[0]
            self.class_dictionary_ = {}
            for index, class_val in enumerate(self.classes_):
                self.class_dictionary_[class_val] = index

            if self.n_classes_ == 1:
                return self

        self.clf_ = SklearnDummyClassifier(
            strategy=self.strategy,
            random_state=self.random_state,
            constant=self.constant,
        )
        self.clf_.fit(None, y)

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        check_is_fitted(self)

        if self.validate:
            # treat case of single class seen in fit
            if self.n_classes_ == 1:
                return np.repeat(
                    list(self.class_dictionary_.keys()), X.shape[0], axis=0
                )

            X = self._validate_data(X=X, reset=False, ensure_min_series_length=1)

        return self.clf_.predict(np.zeros((_num_samples(X), 2)))

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        if self.validate:
            # treat case of single class seen in fit
            if self.n_classes_ == 1:
                return np.repeat([[1]], X.shape[0], axis=0)

            X = self._validate_data(X=X, reset=False, ensure_min_series_length=1)

        return self.clf_.predict_proba(np.zeros((_num_samples(X), 2)))

    def _more_tags(self) -> dict:
        return {
            "X_types": ["3darray", "2darray", "np_list"],
            "equal_length_only": False,
            "no_validation": not self.validate,
            "allow_nan": True,
        }


class DummyRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """DummyRegressor makes predictions that ignore the input features.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    The specific behavior of the baseline is selected with the `strategy`
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`. The predictions, however,
    typically depend on values observed in the `y` parameter passed to `fit`.

    A wrapper for `sklearn.dummy.DummyRegressor` using the tsml interface. Functionally
    identical.

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions.

        - "mean": always predicts the mean of the training set
        - "median": always predicts the median of the training set
        - "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        - "constant": always predicts a constant value that is provided by
          the user.
    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.
    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.
    validate : bool, default=False
        Whether to perform validation checks on X and y.

    See Also
    --------
    DummyClassifier : Classifier that makes predictions using simple rules.
    DummyClusterer : Clusterer that makes predictions using simple rules.

    Examples
    --------
    >>> from tsml.dummy import DummyRegressor
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10,
    ...                              regression_target=True, random_state=0)
    >>> reg = DummyRegressor()
    >>> reg.fit(X, y)
    DummyRegressor(...)
    >>> reg.score(X, y)
    0.0
    """

    def __init__(self, strategy="mean", constant=None, quantile=None, validate=False):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile
        self.validate = validate

        super(DummyRegressor, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The target labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        if self.validate:
            _, y = self._validate_data(
                X=X, y=y, ensure_min_series_length=1, y_numeric=True
            )

        self.reg_ = SklearnDummyRegressor(
            strategy=self.strategy, constant=self.constant, quantile=self.quantile
        )
        self.reg_.fit(None, y)

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted target labels.
        """
        check_is_fitted(self)

        if self.validate:
            X = self._validate_data(X=X, reset=False, ensure_min_series_length=1)

        return self.reg_.predict(np.zeros((_num_samples(X), 2)))

    def _more_tags(self) -> dict:
        return {
            "X_types": ["3darray", "2darray", "np_list"],
            "equal_length_only": False,
            "no_validation": not self.validate,
            "allow_nan": True,
        }


class DummyClusterer(ClusterMixin, BaseTimeSeriesEstimator):
    """DummyClusterer makes predictions that ignore the input features.

    This cluster makes no effort to form reasonable clusters, and is primarily used
    for interface testing. Do not use it for real problems.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`.

    Parameters
    ----------
    strategy : {"single", "unique", "random"}, default="single"
        Strategy to use to generate clusters.
        - "single": all cases are assigned to cluster 0.
        - "unique": all cases are assigned thier own cluster.
        - "random": randomly assigned cases to one of ``n_clusters`` clusters.
    n_clusters : int, default=2
        The number of clusters to generate when ``strategy='random'``.
    validate : bool, default=False
        Whether to perform validation checks on X and y.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness to generate the clusters when ``strategy='random'``
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    See Also
    --------
    DummyClassifier : DummyClassifier is a classifier that makes predictions
    DummyRegressor : Regressor that makes predictions using simple rules.

    Examples
    --------
    >>> from tsml.dummy import DummyClusterer
    >>> from sklearn.metrics import adjusted_rand_score
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> clu = DummyClusterer(strategy="random", random_state=0)
    >>> clu.fit(X)
    DummyClusterer(...)
    >>> adjusted_rand_score(clu.labels_, y)
    0.16
    """

    def __init__(
        self, strategy="single", n_clusters=2, validate=False, random_state=None
    ):
        self.strategy = strategy
        self.n_clusters = n_clusters
        self.validate = validate
        self.random_state = random_state

        super(DummyClusterer, self).__init__()

    def fit(
        self, X: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, None] = None
    ) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The input data.
        y : 1D np.ndarray of shape (n_instances), default=None
            Label placeholder for compatability. Not used.

        Returns
        -------
        self :
            Reference to self.
        """
        if self.validate:
            X = self._validate_data(X=X, ensure_min_series_length=1)

        if self.strategy == "single":
            self.labels_ = np.zeros(len(X), dtype=np.int32)
        elif self.strategy == "unique":
            self.labels_ = np.arange(len(X), dtype=np.int32)
        elif self.strategy == "random":
            rng = check_random_state(self.random_state)
            self.labels_ = rng.randint(self.n_clusters, size=len(X), dtype=np.int32)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Assign clusters labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The input data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Assigned cluster labels.
        """
        check_is_fitted(self)

        if self.validate:
            X = self._validate_data(X=X, reset=False, ensure_min_series_length=1)

        if self.strategy == "single":
            return np.zeros(_num_samples(X), dtype=np.int32)
        elif self.strategy == "unique":
            return np.arange(_num_samples(X), dtype=np.int32)
        elif self.strategy == "random":
            rng = check_random_state(self.random_state)
            return rng.randint(self.n_clusters, size=_num_samples(X), dtype=np.int32)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def _more_tags(self) -> dict:
        return {
            "X_types": ["3darray", "2darray", "np_list"],
            "equal_length_only": False,
            "no_validation": not self.validate,
            "allow_nan": True,
        }
