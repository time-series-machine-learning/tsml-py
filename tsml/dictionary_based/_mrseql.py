"""Multiple Representations Sequence Learning (MrSEQL) Classifier."""

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class MrSEQLClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """
    Multiple Representations Sequence Learning (MrSEQL) Classifier.

    This is a wrapper for the MrSEQLClassifier algorithm from the `mrseql` package.
    MrSEQL is not included in ``all_extras`` as it requires gcc and fftw
    (http://www.fftw.org/index.html) to be installed for Windows and some Linux OS.

    Overview: MrSEQL extends the symbolic sequence classifier (SEQL) to work with
    multiple symbolic representations of time series, using features extracted from the
    SAX and SFA transformations.

    Parameters
    ----------
    seql_mode : "clf" or "fs", default="fs".
        If "fs", trains a logistic regression model with features extracted by SEQL.
        IF "clf", builds an ensemble of SEQL models
    symrep : "sax" or "sfa", or ["sax", "sfa"], default = "sax"
        The symbolic features to extract from the time series.
    custom_config : dict, default=None
        Additional configuration for the symbolic transformations. See the original
        package for details. ``symrep`` will be ignored if used.

    References
    ----------
    .. [1] Le Nguyen, Thach, et al. "Interpretable time series classification using
        linear models and multi-resolution multi-domain symbolic representations."
        Data mining and knowledge discovery 33 (2019): 1183-1222.
    """

    def __init__(self, seql_mode="fs", symrep=("sax"), custom_config=None) -> None:
        self.seql_mode = seql_mode
        self.symrep = symrep
        self.custom_config = custom_config

        _check_optional_dependency("mrseql", "mrseql", self)

        super().__init__()

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

        from mrseql import MrSEQLClassifier

        _X = _convert_data(X)

        self.clf_ = MrSEQLClassifier(
            seql_mode=self.seql_mode,
            symrep=self.symrep,
            custom_config=self.custom_config,
        )
        self.clf_.fit(_X, y)

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

        return self.clf_.predict(_convert_data(X))

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

        return self.clf_.predict_proba(_convert_data(X))

    def _more_tags(self) -> dict:
        return {
            "non_deterministic": True,
            "_xfail_checks": {"check_estimators_pickle": "External failure to pickle."},
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
        return {}


def _convert_data(X):
    column_list = []
    for i in range(X.shape[1]):
        nested_column = (
            pd.DataFrame(X[:, i, :])
            .apply(lambda x: [pd.Series(x, dtype=X.dtype)], axis=1)
            .str[0]
            .rename(str(i))
        )
        column_list.append(nested_column)
    df = pd.concat(column_list, axis=1)
    return df
