# -*- coding: utf-8 -*-
"""Catch22 Classifier.

Pipeline classifier using the Catch22 transformer and an estimator.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["Catch22Classifier", "Catch22Regressor"]

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations.catch22 import Catch22Transformer
from tsml.utils.validation import check_n_jobs


class Catch22Classifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Canonical Time-series Characteristics (catch22) classifier.

    This classifier simply transforms the input data using the Catch22 [1]
    transformer and builds a provided estimator using the transformed data.

    Shorthand for the pipeline `Catch22(outlier_norm, replace_nans) * estimator`

    Parameters
    ----------
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, optional, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, optional, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    estimator_ : ClassifierPipeline
        Catch22Classifier as a ClassifierPipeline, fitted to data internally

    See Also
    --------
    Catch22

    Notes
    -----
    Authors `catch22ForestClassifier <https://github.com/chlubba/sktime-catch22>`_.

    For the Java version, see `tsml <https://github.com/uea-machine-learning/tsml/blob
    /master/src/main/java/tsml/classifiers/hybrids/Catch22Classifier.java>`_.

    References
    ----------
    .. [1] Lubba, Carl H., et al. "catch22: Canonical time-series characteristics."
        Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.
        https://link.springer.com/article/10.1007/s10618-019-00647-x
    """

    def __init__(
        self,
        features="all",
        catch24=False,
        outlier_norm=False,
        replace_nans=True,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        super(Catch22Classifier, self).__init__()

    def fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=3
        )

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary_[classVal] = index

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = Catch22Transformer(
            features=self.features,
            catch24=self.catch24,
            outlier_norm=self.outlier_norm,
            replace_nans=self.replace_nans,
            n_jobs=self._n_jobs,
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

    def predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)

        return self._estimator.predict(self._transformer.transform(X))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)

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
        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestClassifier(n_estimators=10),
                "features": (
                    "Mean",
                    "DN_HistogramMode_5",
                    "SB_BinaryStats_mean_longstretch1",
                ),
                "catch24": True,
                "replace_nans": True,
                "outlier_norm": True,
            }
        else:
            return {
                "estimator": RandomForestClassifier(n_estimators=2),
                "features": (
                    "Mean",
                    "DN_HistogramMode_5",
                    "SB_BinaryStats_mean_longstretch1",
                ),
                "catch24": True,
                "replace_nans": True,
            }


class Catch22Regressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Canonical Time-series Characteristics (catch22) classifier.

    This classifier simply transforms the input data using the Catch22 [1]
    transformer and builds a provided estimator using the transformed data.

    Shorthand for the pipeline `Catch22(outlier_norm, replace_nans) * estimator`

    Parameters
    ----------
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, optional, default=True
        Replace NaN or inf values from the Catch22 transform with 0.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn RandomForestClassifier(n_estimators=200)
    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, optional, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    estimator_ : ClassifierPipeline
        Catch22Classifier as a ClassifierPipeline, fitted to data internally

    See Also
    --------
    Catch22

    Notes
    -----
    Authors `catch22ForestClassifier <https://github.com/chlubba/sktime-catch22>`_.

    For the Java version, see `tsml <https://github.com/uea-machine-learning/tsml/blob
    /master/src/main/java/tsml/classifiers/hybrids/Catch22Classifier.java>`_.

    References
    ----------
    .. [1] Lubba, Carl H., et al. "catch22: Canonical time-series characteristics."
        Data Mining and Knowledge Discovery 33.6 (2019): 1821-1852.
        https://link.springer.com/article/10.1007/s10618-019-00647-x
    """

    def __init__(
        self,
        features="all",
        catch24=False,
        outlier_norm=False,
        replace_nans=True,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.estimator = estimator

        self.n_jobs = n_jobs
        self.random_state = random_state

        super(Catch22Regressor, self).__init__()

    def fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" and sets is_fitted flag to True.
        """
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = Catch22Transformer(
            features=self.features,
            catch24=self.catch24,
            outlier_norm=self.outlier_norm,
            replace_nans=self.replace_nans,
            n_jobs=self._n_jobs,
        )

        self._estimator = _clone_estimator(
            RandomForestRegressor(n_estimators=200)
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

    def predict(self, X) -> np.ndarray:
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)

        return self._estimator.predict(self._transformer.transform(X))

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
        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestRegressor(n_estimators=10),
                "features": (
                    "Mean",
                    "DN_HistogramMode_5",
                    "SB_BinaryStats_mean_longstretch1",
                ),
                "catch24": True,
                "replace_nans": True,
                "outlier_norm": True,
            }
        else:
            return {
                "estimator": RandomForestRegressor(n_estimators=2),
                "features": (
                    "Mean",
                    "DN_HistogramMode_5",
                    "SB_BinaryStats_mean_longstretch1",
                ),
                "catch24": True,
                "replace_nans": True,
            }
