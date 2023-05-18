# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class MrSQMClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """
    Wrapper for https://github.com/mlgig/mrsqm.
    """

    def __init__(
        self,
        strat="RS",
        features_per_rep=500,
        selection_per_rep=2000,
        nsax=0,
        nsfa=5,
        sfa_norm=True,
        custom_config=None,
        random_state=None,
    ):
        self.strat = strat
        self.features_per_rep = features_per_rep
        self.selection_per_rep = selection_per_rep
        self.nsax = nsax
        self.nsfa = nsfa
        self.sfa_norm = sfa_norm
        self.custom_config = custom_config
        self.random_state = random_state

        _check_optional_dependency("mrsqm", "mrsqm", self)

        super(MrSQMClassifier, self).__init__()

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

        from mrsqm import MrSQMClassifier

        self.clf_ = MrSQMClassifier(
            strat=self.strat,
            features_per_rep=self.features_per_rep,
            selection_per_rep=self.selection_per_rep,
            nsax=self.nsax,
            nsfa=self.nsfa,
            sfa_norm=self.sfa_norm,
            custom_config=self.custom_config,
            random_state=self.random_state,
        )
        self.clf_.fit(_convert_data(X), y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        return self.clf_.predict(_convert_data(X))

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        return self.clf_.predict_proba(_convert_data(X))

    def _more_tags(self):
        return {
            "non_deterministic": True,
            "_xfail_checks": {"check_estimators_pickle": "External failure to pickle."},
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
            "features_per_rep": 50,
            "selection_per_rep": 200,
            "nsax": 1,
            "nsfa": 1,
        }


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
