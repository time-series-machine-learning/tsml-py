# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class RandomShapeletForestClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """
    Wrapper for https://github.com/wildboar-foundation/wildboar RSF implementation.
    """

    def __init__(
        self,
        n_estimators=100,
        n_shapelets=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="entropy",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        class_weight=None,
        n_jobs=None,
        random_state=None,
    ):
        self.n_shapelets = n_shapelets
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.alpha = alpha
        self.metric = metric
        self.metric_params = metric_params
        self.criterion = criterion
        self.oob_score = oob_score
        self.bootstrap = bootstrap
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        _check_optional_dependency("wildboar", "wildboar", self)

        super(RandomShapeletForestClassifier, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = (
            X.shape if X.ndim == 3 else (X.shape[0], 1, X.shape[1])
        )
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        from wildboar.ensemble import ShapeletForestClassifier

        self.clf_ = ShapeletForestClassifier(
            n_shapelets=self.n_shapelets,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            alpha=self.alpha,
            metric=self.metric,
            metric_params=self.metric_params,
            criterion=self.criterion,
            oob_score=self.oob_score,
            bootstrap=self.bootstrap,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            n_jobs=self._n_jobs,
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
        X = self._convert_X(X)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf_.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf_.predict_proba(X)

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
            "n_estimators": 2,
        }


class RandomShapeletForestRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """
    Wrapper for https://github.com/wildboar-foundation/wildboar RSF implementation.
    """

    def __init__(
        self,
        n_estimators=100,
        n_shapelets=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        metric="euclidean",
        metric_params=None,
        criterion="squared_error",
        oob_score=False,
        bootstrap=True,
        warm_start=False,
        n_jobs=None,
        random_state=None,
    ):
        self.n_shapelets = n_shapelets
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.alpha = alpha
        self.metric = metric
        self.metric_params = metric_params
        self.criterion = criterion
        self.oob_score = oob_score
        self.bootstrap = bootstrap
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state

        _check_optional_dependency("wildboar", "wildboar", self)

        super(RandomShapeletForestRegressor, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        self._n_jobs = check_n_jobs(self.n_jobs)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        from wildboar.ensemble import ShapeletForestRegressor

        self.reg_ = ShapeletForestRegressor(
            n_shapelets=self.n_shapelets,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
            alpha=self.alpha,
            metric=self.metric,
            metric_params=self.metric_params,
            criterion=self.criterion,
            oob_score=self.oob_score,
            bootstrap=self.bootstrap,
            warm_start=self.warm_start,
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        self.reg_.fit(X, y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.reg_.predict(X)

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
            "n_estimators": 2,
        }
