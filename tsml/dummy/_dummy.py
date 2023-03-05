# -*- coding: utf-8 -*-
"""Dummy time series estimators."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["DummyClassifier", "DummyRegressor", "DummyClusterer"]

import numpy as np
from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier
from sklearn.dummy import DummyRegressor as SklearnDummyRegressor
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

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

        * "most_frequent": the `predict` method always returns the most
          frequent class label in the observed `y` argument passed to `fit`.
          The `predict_proba` method returns the matching one-hot encoded
          vector.
        * "prior": the `predict` method always returns the most frequent
          class label in the observed `y` argument passed to `fit` (like
          "most_frequent"). ``predict_proba`` always returns the empirical
          class distribution of `y` also known as the empirical class prior
          distribution.
        * "stratified": the `predict_proba` method randomly samples one-hot
          vectors from a multinomial distribution parametrized by the empirical
          class prior probabilities.
          The `predict` method returns the class label which got probability
          one in the one-hot vector of `predict_proba`.
          Each sampled row of both methods is therefore independent and
          identically distributed.
        * "uniform": generates predictions uniformly at random from the list
          of unique classes observed in `y`, i.e. each class has equal
          probability.
        * "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness to generate the predictions when
        ``strategy='stratified'`` or ``strategy='uniform'``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    constant : int or str or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    See Also
    --------
    DummyRegressor : Regressor that makes predictions using simple rules.

    Examples
    --------
    >>> from tsml.dummy import DummyClassifier
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X_train, y_train = load_minimal_chinatown(split="train")
    >>> X_test, y_test = load_minimal_chinatown(split="test")
    >>> clf = DummyClassifier(strategy="most_frequent")
    >>> clf.fit(X_train, y_train)
    DummyClassifier(strategy='most_frequent')
    >>> clf.score(X_test, y_test)
    0.5
    """

    def __init__(self, strategy="prior", random_state=None, constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

        super(DummyClassifier, self).__init__()

    def fit(self, X, y):
        """"""
        X, y = self._validate_data(X=X, y=y)

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary_[classVal] = index

        if len(self.classes_) == 1:
            return self

        self._clf = SklearnDummyClassifier(
            strategy=self.strategy,
            random_state=self.random_state,
            constant=self.constant,
        )
        self._clf.fit(np.zeros(X.shape), y)

        return self

    def predict(self, X) -> np.ndarray:
        """"""
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)

        return self._clf.predict(np.zeros(X.shape))

    def predict_proba(self, X) -> np.ndarray:
        """"""
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)

        return self._clf.predict_proba(np.zeros(X.shape))


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

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.
    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.
    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    See Also
    --------
    DummyClassifier : Classifier that makes predictions using simple rules.

    Examples
    --------
    >>> from tsml.dummy import DummyRegressor
    >>> from tsml.datasets import load_minimal_gas_prices
    >>> X_train, y_train = load_minimal_gas_prices(split="train")
    >>> X_test, y_test = load_minimal_gas_prices(split="test")
    >>> reg = DummyRegressor()
    >>> reg.fit(X_train, y_train)
    DummyRegressor()
    >>> reg.score(X_test, y_test)
    -0.07184048625633688
    """

    def __init__(self, strategy="mean", constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

        super(DummyRegressor, self).__init__()

    def fit(self, X, y):
        """"""
        X, y = self._validate_data(X=X, y=y)

        self._reg = SklearnDummyRegressor(
            strategy=self.strategy, constant=self.constant, quantile=self.quantile
        )
        self._reg.fit(np.zeros(X.shape), y)

        return self

    def predict(self, X):
        """"""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)

        return self._reg.predict(np.zeros(X.shape))


class DummyClusterer(ClusterMixin, BaseTimeSeriesEstimator):
    """DummyRegressor makes predictions that ignore the input features.

    This cluster makes no effort to form reasonable clusters, and is primarily used
    for interface testing. Do not use it for real problems.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`.

    todo example adjusted_rand_score

    Examples
    --------
    >>> from tsml.dummy import DummyClusterer
    >>> from tsml.datasets import load_minimal_chinatown
    >>> from sklearn.metrics import adjusted_rand_score
    >>> X_train, y_train = load_minimal_chinatown(split="train")
    >>> X_test, y_test = load_minimal_chinatown(split="test")
    >>> clu = DummyClusterer(strategy="random", random_state=0)
    >>> clu.fit(X_train)
    DummyClusterer(random_state=0, strategy='random')
    >>> adjusted_rand_score(clu.labels_, y_train)
    0.2087729039422543
    >>> adjusted_rand_score(clu.predict(X_test), y_test)
    0.2087729039422543
    """

    def __init__(self, strategy="single", n_clusters=2, random_state=None):
        self.strategy = strategy
        self.n_clusters = n_clusters
        self.random_state = random_state

        super(DummyClusterer, self).__init__()

    def fit(self, X, y=None):
        """"""
        X = self._validate_data(X=X)

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

    def predict(self, X):
        """"""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)

        if self.strategy == "single":
            return np.zeros(len(X), dtype=np.int32)
        elif self.strategy == "unique":
            return np.arange(len(X), dtype=np.int32)
        elif self.strategy == "random":
            rng = check_random_state(self.random_state)
            return rng.randint(self.n_clusters, size=len(X), dtype=np.int32)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {}
