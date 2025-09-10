"""A tsml wrapper for sklearn classifiers."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = [
    "SklearnToTsmlClassifier",
    "SklearnToTsmlClusterer",
    "SklearnToTsmlRegressor",
]

import numpy as np
from aeon.base._base import _clone_estimator
from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator


class SklearnToTsmlClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators to use the tsml base class."""

    def __init__(
        self,
        classifier=None,
        pad_unequal=False,
        concatenate_channels=False,
        clone_estimator=True,
        random_state=None,
    ):
        self.classifier = classifier
        self.pad_unequal = pad_unequal
        self.concatenate_channels = concatenate_channels
        self.clone_estimator = clone_estimator
        self.random_state = random_state

        super().__init__()

    def fit(self, X, y):
        """Wrap fit."""
        if self.classifier is None:
            raise ValueError("Classifier not set")

        X, y = self._validate_data(
            X=X,
            y=y,
            ensure_univariate=not self.concatenate_channels,
            ensure_equal_length=not self.pad_unequal,
        )
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        check_classification_targets(y)
        self.classes_ = np.unique(y)

        self._classifier = (
            _clone_estimator(self.classifier, self.random_state)
            if self.clone_estimator
            else self.classifier
        )
        self._classifier.fit(X, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Wrap predict."""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        return self._classifier.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Wrap predict_proba."""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        return self._classifier.predict_proba(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "equal_length_only": (False if self.pad_unequal else True),
            "univariate_only": False if self.concatenate_channels else True,
        }

    @classmethod
    def get_test_params(cls, parameter_set: str | None = None) -> dict | list[dict]:
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
        from sklearn.ensemble import RandomForestClassifier

        return {"classifier": RandomForestClassifier(n_estimators=5)}


class SklearnToTsmlClusterer(ClusterMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators to use the tsml base class."""

    def __init__(
        self,
        clusterer=None,
        pad_unequal=False,
        concatenate_channels=False,
        clone_estimator=True,
        random_state=None,
    ):
        self.clusterer = clusterer
        self.pad_unequal = pad_unequal
        self.concatenate_channels = concatenate_channels
        self.clone_estimator = clone_estimator
        self.random_state = random_state

        super().__init__()

    def fit(self, X, y=None):
        """Wrap fit."""
        if self.clusterer is None:
            raise ValueError("Clusterer not set")

        X = self._validate_data(
            X=X,
            ensure_univariate=not self.concatenate_channels,
            ensure_equal_length=not self.pad_unequal,
        )
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        self._clusterer = (
            _clone_estimator(self.clusterer, self.random_state)
            if self.clone_estimator
            else self.clusterer
        )
        self._clusterer.fit(X, y)

        self.labels_ = self._clusterer.labels_

        return self

    def predict(self, X) -> np.ndarray:
        """Wrap predict."""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        return self._clusterer.predict(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "equal_length_only": (False if self.pad_unequal else True),
            "univariate_only": False if self.concatenate_channels else True,
        }

    @classmethod
    def get_test_params(cls, parameter_set: str | None = None) -> dict | list[dict]:
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
        from sklearn.cluster import KMeans

        return {"clusterer": KMeans(n_clusters=2, max_iter=5)}


class SklearnToTsmlRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Wrapper for sklearn estimators to use the tsml base class."""

    def __init__(
        self,
        regressor=None,
        pad_unequal=False,
        concatenate_channels=False,
        clone_estimator=True,
        random_state=None,
    ):
        self.regressor = regressor
        self.pad_unequal = pad_unequal
        self.concatenate_channels = concatenate_channels
        self.clone_estimator = clone_estimator
        self.random_state = random_state

        super().__init__()

    def fit(self, X, y):
        """Wrap fit."""
        if self.regressor is None:
            raise ValueError("Regressor not set")

        X, y = self._validate_data(
            X=X,
            y=y,
            ensure_univariate=not self.concatenate_channels,
            ensure_equal_length=not self.pad_unequal,
        )
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        self._regressor = (
            _clone_estimator(self.regressor, self.random_state)
            if self.clone_estimator
            else self.regressor
        )
        self._regressor.fit(X, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Wrap predict."""
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(
            X,
            pad_unequal=self.pad_unequal,
            concatenate_channels=self.concatenate_channels,
        )

        return self._regressor.predict(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "equal_length_only": (False if self.pad_unequal else True),
            "univariate_only": False if self.concatenate_channels else True,
        }

    @classmethod
    def get_test_params(cls, parameter_set: str | None = None) -> dict | list[dict]:
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
        from sklearn.ensemble import RandomForestRegressor

        return {"regressor": RandomForestRegressor(n_estimators=5)}
