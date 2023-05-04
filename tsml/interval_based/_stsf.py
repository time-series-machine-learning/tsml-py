# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]
__all__ = ["STSFClassifier", "RSTSFClassifier"]

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.interval_based._base import BaseIntervalForest
from tsml.transformations import (
    ARCoefficientTransformer,
    FunctionTransformer,
    PeriodogramTransformer,
    SupervisedIntervalTransformer,
)
from tsml.utils.numba_functions.general import first_order_differences_3d
from tsml.utils.numba_functions.stats import (
    row_count_above_mean,
    row_count_mean_crossing,
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_slope,
    row_std,
)
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class STSFClassifier(ClassifierMixin, BaseIntervalForest):
    """TODO."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        min_interval_length=3,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        use_pyfftw=True,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.use_pyfftw = use_pyfftw
        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)

        series_transformers = [
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(use_pyfftw=use_pyfftw),
        ]

        interval_features = [
            row_mean,
            row_std,
            row_slope,
            row_median,
            row_iqr,
            row_numba_min,
            row_numba_max,
        ]

        super(STSFClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="supervised",
            n_intervals=1,
            min_interval_length=min_interval_length,
            max_interval_length=np.inf,
            interval_features=interval_features,
            series_transformers=series_transformers,
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

    def _more_tags(self):
        return {
            "optional_dependency": self.use_pyfftw,
        }


class RSTSFClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    def __init__(
        self,
        n_estimators=200,
        n_intervals=50,
        min_interval_length=3,
        use_pyfftw=True,
        random_state=None,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.use_pyfftw = use_pyfftw
        self.random_state = random_state
        self.n_jobs = n_jobs

        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)
        _check_optional_dependency("statsmodels", "statsmodels", self)

        super(RSTSFClassifier, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=5
        )
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary_[classVal] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._series_transformers = [
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(use_pyfftw=self.use_pyfftw),
            ARCoefficientTransformer(replace_nan=True),
        ]

        transforms = [X] + [t.fit_transform(X) for t in self._series_transformers]

        Xt = np.empty((X.shape[0], 0))
        self._transformers = []
        for i, t in enumerate(transforms):
            si = SupervisedIntervalTransformer(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
                randomised_split_point=True,
            )
            Xt = np.hstack((Xt, si.fit_transform(t, y)))
            self._transformers.append(si)

        self.clf_ = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion="entropy",
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        self.clf_.fit(Xt, y)

        return self

    def predict(self, X):
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        Xt = self._predict_transform(X)
        return self.clf_.predict(Xt)

    def predict_proba(self, X):
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        Xt = self._predict_transform(X)
        return self.clf_.predict_proba(Xt)

    def _predict_transform(self, X):
        X = self._validate_data(X=X, ensure_min_series_length=5, reset=False)
        X = self._convert_X(X)

        transforms = [X] + [t.transform(X) for t in self._series_transformers]

        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            si = self._transformers[i]
            Xt = np.hstack((Xt, si.transform(t)))

        return Xt

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
            "n_intervals": 2,
        }

    def _more_tags(self):
        return {
            "optional_dependency": True,
        }
