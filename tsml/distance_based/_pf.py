# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class ProximityForestClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """
    Wrapper for https://github.com/wildboar-foundation/wildboar PF implementation.
    """
    def __init__(
        self,
        n_estimators=100,
        n_pivot=1,
        pivot_sample="label",
        metric_sample="weighted",
        metric_factories="default",
        oob_score=False,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0,
        criterion="entropy",
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        class_weight=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.n_pivot = n_pivot
        self.pivot_sample = pivot_sample
        self.metric_sample = metric_sample
        self.metric_factories = metric_factories
        self.oob_score = oob_score
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.random_state = random_state

        _check_optional_dependency("wildboar", "wildboar", self)

        super(ProximityForestClassifier, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = (
            X.shape if X.ndim == 3 else (X.shape[0], 1, X.shape[1])
        )
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary_[classVal] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        from wildboar.ensemble import ProximityForestClassifier

        self.clf_ = ProximityForestClassifier(
            n_estimators=self.n_estimators,
            n_pivot=self.n_pivot,
            pivot_sample=self.pivot_sample,
            metric_sample=self.metric_sample,
            metric_factories=self.metric_factories,
            oob_score=self.oob_score,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            criterion=self.criterion,
            bootstrap=self.bootstrap,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.clf_.fit(X, y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf_.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf_.predict_proba(X)
