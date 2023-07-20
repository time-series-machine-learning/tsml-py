"""Random Shapelet Forest (RSF) estimators."""
from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class RandomShapeletForestClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Random Shapelet Forest (RSF) Classifier.

    Wrapper for the wildboar RSF implementation.
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
        random_state=None,
        n_jobs=None,
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
        self.random_state = random_state
        self.n_jobs = n_jobs

        _check_optional_dependency("wildboar", "wildboar", self)

        super(RandomShapeletForestClassifier, self).__init__()

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
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = (
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
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self.clf_.fit(X, y)

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

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf_.predict(X)

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

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.clf_.predict_proba(X)

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
        }


class RandomShapeletForestRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Random Shapelet Forest (RSF) Regressor.

    Wrapper for the wildboar RSF implementation.
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
        random_state=None,
        n_jobs=None,
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
        self.random_state = random_state
        self.n_jobs = n_jobs

        _check_optional_dependency("wildboar", "wildboar", self)

        super(RandomShapeletForestRegressor, self).__init__()

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
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2, y_numeric=True)
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
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self.reg_.fit(X, y)

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

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        if X.ndim == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], X.shape[2]))

        return self.reg_.predict(X)

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
        }
