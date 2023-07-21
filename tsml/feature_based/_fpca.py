"""FPCA pipeline estimators.

Classical Scalar on Function approach that allows transforming
via B-spline if desired.
"""

__author__ = ["dguijo", "MatthewMiddlehurst"]
__all__ = ["FPCAClassifier", "FPCARegressor"]

from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations import FPCATransformer
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class FPCAClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Functional Principal Component Analysis pipeline classifier.

    This classifier simply transforms the input data using the FPCA
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    n_components: int, default=10
        Number of principal components to keep from functional principal component
        analysis.
    centering: bool, default=True
        Set to ``False`` when the functional data is already known to be centered
        and there is no need to center it. Otherwise, the mean of the functional
        data object is calculated and the data centered before fitting.
    bspline: bool, default=False
        Set to ``True`` to use a B-spline basis for the functional principal
        component analysis.
    n_basis: int, default=None
        Number of functions in the basis. Only used if `bspline` is `True`.
    order: int, default=None
        Order of the splines. One greater than their degree. Only used if
        `bspline` is `True`.
    estimator : sklearn classifier, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn LogisticRegression.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

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

    See Also
    --------
    FPCATransformer
    FPCARegressor
    """

    def __init__(
        self,
        n_components=10,
        centering=True,
        bspline=False,
        n_basis=None,
        order=None,
        estimator=None,
        random_state=None,
        n_jobs=1,
    ):
        self.n_components = n_components
        self.centering = centering
        self.bspline = bspline
        self.n_basis = n_basis
        self.order = order
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs

        _check_optional_dependency("scikit-fda", "skfda", self)

        super(FPCAClassifier, self).__init__()

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

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
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
            bspline=self.bspline,
            n_basis=self.n_basis,
            order=self.order,
        )

        self._estimator = _clone_estimator(
            LogisticRegression() if self.estimator is None else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y).reshape((self.n_instances_, -1))
        self._estimator.fit(X_t, y)

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

        return self._estimator.predict(
            self._transformer.transform(X).reshape((X.shape[0], -1))
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
            "n_components": 2,
        }


class FPCARegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Scalar on Function Regression using Functional Principal Component Analysis.

    This regressor simply transforms the input data using the FPCA
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    n_components: int, default=10
        Number of principal components to keep from functional principal component
        analysis.
    centering: bool, default=True
        Set to ``False`` when the functional data is already known to be centered
        and there is no need to center it. Otherwise, the mean of the functional
        data object is calculated and the data centered before fitting.
    bspline: bool, default=False
        Set to ``True`` to use a B-spline basis for the functional principal
        component analysis.
    n_basis: int, default=None
        Number of functions in the basis. Only used if `bspline` is `True`.
    order: int, default=None
        Order of the splines. One greater than their degree. Only used if
        `bspline` is `True`.
    estimator : sklearn regressor, optional, default=None
        An sklearn estimator to be built using the transformed data.
        Defaults to sklearn LinearRegression.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.

    See Also
    --------
    FPCATransformer
    FPCAClassifier
    """

    def __init__(
        self,
        n_components=10,
        centering=True,
        bspline=False,
        n_basis=None,
        order=None,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_components = n_components
        self.centering = centering
        self.bspline = bspline
        self.n_basis = n_basis
        self.order = order
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state

        _check_optional_dependency("scikit-fda", "skfda", self)

        super(FPCARegressor, self).__init__()

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
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = FPCATransformer(
            n_components=self.n_components,
            centering=self.centering,
            bspline=self.bspline,
            n_basis=self.n_basis,
            order=self.order,
        )

        self._estimator = _clone_estimator(
            LinearRegression() if self.estimator is None else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y).reshape((X.shape[0], -1))
        self._estimator.fit(X_t, y)

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

        return self._estimator.predict(
            self._transformer.transform(X).reshape((X.shape[0], -1))
        )

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
            "n_components": 2,
        }
