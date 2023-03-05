# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]
__all__ = ["CIFClassifier", "CIFRegressor", "DrCIFClassifier", "DrCIFRegressor"]

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tsml.interval_based._base import BaseIntervalForest
from tsml.transformations.catch22 import Catch22Transformer
from tsml.utils.numba_functions.stats import row_mean, row_slope, row_std
from tsml.vector import CITClassifier


class CIFClassifier(ClassifierMixin, BaseIntervalForest):
    """TODO."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        att_subsample_size=8,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(criterion="entropy")

        if isinstance(base_estimator, CITClassifier):
            replace_nan = "nan"
        else:
            replace_nan = "zero"

        interval_features = [
            Catch22Transformer(outlier_norm=True),
            row_mean,
            row_std,
            row_slope,
        ]

        super(CIFClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=None,
            att_subsample_size=att_subsample_size,
            replace_nan=replace_nan,
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
            "n_intervals": 2,
            "att_subsample_size": 2,
        }


class CIFRegressor(ClassifierMixin, BaseIntervalForest):
    """TODO."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        att_subsample_size=8,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        if base_estimator is None:
            base_estimator = DecisionTreeRegressor()

        interval_features = [
            Catch22Transformer(outlier_norm=True),
            row_mean,
            row_std,
            row_slope,
        ]

        super(CIFRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=None,
            att_subsample_size=att_subsample_size,
            replace_nan="zero",
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
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
            "n_estimators": 2,
            "n_intervals": 2,
            "att_subsample_size": 2,
        }


class DrCIFClassifier(ClassifierMixin, BaseIntervalForest):
    """TODO."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        n_intervals=None,
        min_interval_length=3,
        max_interval_length=0.5,
        att_subsample_size=10,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        # CIT
        # nans if CIT

        # if n_intervals is None:
        #     n_intervals =
        # ((4, "sqrt-div"), (4, "sqrt-div"), (4, "sqrt-div"))

        # interval_features = [Catch22(outlier_norm=True), None, None, None]

        # check_classification_targets(y)

        super(DrCIFClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=0,
            series_transformers=0,
            att_subsample_size=att_subsample_size,
            replace_nan=0,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
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
            "n_estimators": 2,
            "n_intervals": 2,
            "att_subsample_size": 2,
        }


class DrCIFRegressor(RegressorMixin, BaseIntervalForest):
    """TODO."""

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        n_intervals=None,
        min_interval_length=3,
        max_interval_length=0.5,
        att_subsample_size=10,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        # CIT
        # nans if CIT

        # if n_intervals is None:
        #     n_intervals =
        # ((4, "sqrt-div"), (4, "sqrt-div"), (4, "sqrt-div"))

        # interval_features = [Catch22(outlier_norm=True), None, None, None]

        super(DrCIFRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method="random",
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=0,
            series_transformers=0,
            att_subsample_size=att_subsample_size,
            replace_nan=0,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
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
            "n_estimators": 2,
            "n_intervals": 2,
            "att_subsample_size": 2,
        }
