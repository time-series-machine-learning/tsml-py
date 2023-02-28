# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]
__all__ = ["STSFClassifier", "RSTSFClassifier"]

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.tree import ExtraTreeClassifier

from tsml.interval_based._base import BaseIntervalForest


class STSFClassifier(ClassifierMixin, BaseIntervalForest):
    """TODO."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        min_interval_length=3,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        # min interval length
        # check defaults for others

        super(STSFClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="supervised",
            n_intervals=1,
            min_interval_length=min_interval_length,
            max_interval_length=0,
            interval_features=0,
            series_transformers=0,
            att_subsample_size=None,
            replace_nan=0,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    def predict_proba(self, X):
        return self._predict_proba(X)

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


class RSTSFClassifier(ClassifierMixin, BaseIntervalForest):
    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        n_intervals=50,
        min_interval_length=3,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        # min interval length
        # check defaults for others

        # per_X = _getPeriodogramRepr(X)
        # diff_X = np.diff(X)
        # ar_X = _ar_coefs(X)
        # ar_X[np.isnan(ar_X)] = 0

        def _getPeriodogramRepr(X):
            nfeats = X.shape[1]
            fft_object = pyfftw.builders.fft(X)
            per_X = np.abs(fft_object())
            return per_X[:, : int(nfeats / 2)]

        def _ar_coefs(X):
            X_transform = []
            lags = int(12 * (X.shape[1] / 100.0) ** (1 / 4.0))
            for i in range(X.shape[0]):
                coefs, _ = burg(X[i, :], order=lags)
                X_transform.append(coefs)
            return np.array(X_transform)

        ExtraTreeClassifier(
            criterion="entropy",
            class_weight="balanced",
            max_features="sqrt",
        )

        super(RSTSFClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random-supervised",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=0,
            interval_features=0,
            series_transformers=0,
            att_subsample_size=None,
            replace_nan=0,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    def predict_proba(self, X):
        return self._predict_proba(X)

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
