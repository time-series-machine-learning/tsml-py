# -*- coding: utf-8 -*-
"""FPCRegressor.

Classical Scalar on Function Regression approach that allows transforming
via B-spline if desired.
"""

__author__ = ["dguijo", "MatthewMiddlehurst"]
__all__ = ["FPCAClassifier", "FPCARegressor"]

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations import FPCATransformer
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class FPCAClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Scalar on Function Regression using Functional Principal Component Analysis."""

    def __init__(
        self,
        n_components=10,
        centering=True,
        regularization=None,
        components_basis=None,
        bspline=False,
        n_basis=None,
        order=None,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.components_basis = components_basis
        self.bspline = bspline
        self.n_basis = n_basis
        self.order = order
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state

        _check_optional_dependency("scikit-fda", "skfda", self)

        super(FPCAClassifier, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
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

        self._transformer = FPCATransformer(
            n_components=self.n_components,
            centering=self.centering,
            regularization=self.regularization,
            components_basis=self.components_basis,
            bspline=self.bspline,
            n_basis=self.n_basis,
            order=self.order,
        )

        self._estimator = _clone_estimator(
            LogisticRegression(fit_intercept=True)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y).reshape((self.n_instances_, -1))
        self._estimator.fit(X_t, y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        return self._estimator.predict(
            self._transformer.transform(X).reshape((X.shape[0], -1))
        )

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(
                self._transformer.transform(X).reshape((X.shape[0], -1))
            )
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(
                self._transformer.transform(X).reshape((X.shape[0], -1))
            )
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
        return {
            "n_components": 2,
        }


class FPCARegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Scalar on Function Regression using Functional Principal Component Analysis."""

    def __init__(
        self,
        n_components=10,
        centering=True,
        regularization=None,
        components_basis=None,
        bspline=False,
        n_basis=None,
        order=None,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_components = n_components
        self.centering = centering
        self.regularization = regularization
        self.components_basis = components_basis
        self.bspline = bspline
        self.n_basis = n_basis
        self.order = order
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state

        _check_optional_dependency("scikit-fda", "skfda", self)

        super(FPCARegressor, self).__init__()

    def fit(self, X, y=None):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = FPCATransformer(
            n_components=self.n_components,
            centering=self.centering,
            regularization=self.regularization,
            components_basis=self.components_basis,
            bspline=self.bspline,
            n_basis=self.n_basis,
            order=self.order,
        )

        self._estimator = _clone_estimator(
            LinearRegression(fit_intercept=True)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y).reshape((X.shape[0], -1))
        self._estimator.fit(X_t, y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        return self._estimator.predict(
            self._transformer.transform(X).reshape((X.shape[0], -1))
        )

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
            "n_components": 2,
        }
