# -*- coding: utf-8 -*-
""""""

__author__ = ["TonyBagnall", "patrickzib", "MatthewMiddlehurst"]
__all__ = ["MPDistClassifier"]

import numpy as np
import stumpy
from sklearn.base import ClassifierMixin
from sklearn.metrics import pairwise
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import check_n_jobs


class MPDistClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """MPDist 1-NN classifier-adaptor."""

    def __init__(self, window=10, n_jobs=1):
        self.window = window
        self.n_jobs = n_jobs

        super(MPDistClassifier, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.series_length_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._X_train = X.astype(np.float64)
        self._y_train = y

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        window = (
            self.window if self.window >= 1 else int(self.window * self.series_length_)
        )

        distance_matrix = pairwise.pairwise_distances(
            X.astype(np.float64),
            self._X_train,
            metric=(lambda x, y: stumpy.mpdist(x, y, window)),
            n_jobs=self._n_jobs,
        )

        return self._y_train[np.argmin(distance_matrix, axis=1)]

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "optional_dependency": True,
        }

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

        return {
            "window": 0.8,
        }
