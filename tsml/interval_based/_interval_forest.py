"""Configurable interval forest estimators."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["IntervalForestClassifier", "IntervalForestRegressor"]

from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from tsml.interval_based._base import BaseIntervalForest


class IntervalForestClassifier(ClassifierMixin, BaseIntervalForest):
    """Configurable interval extracting forest classifier.

    Extracts multiple phase-dependent intervals from time series data and builds a
    base classifier on summary statistic extracted from each interval. Forms and
    ensemble of these classifiers.

    Allows the implementation of classifiers along the lines of [1][2][3]
    which extract intervals and create an ensemble from the subsequent features.

    By default, uses a configuration similar to TimeSeriesFroest [1].

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    interval_selection_method : "random", "supervised" or "random-supervised",
            default="random"
        The interval selection transformer to use.
            - "random" uses a RandomIntervalTransformer.
            - "supervised" uses a SupervisedIntervalTransformer.
            - "random-supervised" uses a SupervisedIntervalTransformer with
                randomised elements.
    n_intervals : int, str, list or tuple, default="sqrt"
        Number of intervals to extract per tree for each series_transformers series.

        An int input will extract that number of intervals from the series, while a str
        input will return a function of the series length (may differ per
        series_transformers output) to extract that number of intervals.
        Valid str inputs are:
            - "sqrt": square root of the series length.
            - "sqrt-div": sqrt of series length divided by the number
                of series_transformers.

        A list or tuple of ints and/or strs will extract the number of intervals using
        the above rules and sum the results for the final n_intervals. i.e. [4, "sqrt"]
        will extract sqrt(n_timepoints) + 4 intervals.

        Different number of intervals for each series_transformers series can be
        specified using a nested list or tuple. Any list or tuple input containing
        another list or tuple must be the same length as the number of
        series_transformers.

        While random interval extraction will extract the n_intervals intervals total
        (removing duplicates), supervised intervals will run the supervised extraction
        process n_intervals times, returning more intervals than specified.
    min_interval_length : int, float, list, or tuple, default=3
        Minimum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the minimum interval length.

        Different minimum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.
    max_interval_length : int, float, list, or tuple, default=np.inf
        Maximum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the maximum interval length.

        Different maximum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.

        Ignored for supervised interval_selection_method inputs.
    interval_features : TransformerMixin, callable, list, tuple, or None, default=None
        The features to extract from the intervals using transformers or callable
        functions. If None, use the mean, standard deviation, and slope of the series.

        Both transformers and functions should be able to take a 2D np.ndarray input.
        Functions should output a 1d array (the feature for each series), and
        transformers should output a 2d array where rows are the features for each
        series. A list or tuple of transformers and/or functions will extract all
        features and concatenate the output.

        Different features for each series_transformers series can be specified using a
        nested list or tuple. Any list or tuple input containing another list or tuple
        must be the same length as the number of series_transformers.
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals. If None,
        use the series as is.

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    att_subsample_size : int, float, list, tuple or None, default=None
        The number of attributes to subsample for each estimator. If None, use all

        If int, use that number of attributes for all estimators. If float, use that
        proportion of attributes for all estimators.

        Different subsample sizes for each series_transformers series can be specified
        using a list or tuple. Any list or tuple input must be the same length as the
        number of series_transformers.
    replace_nan : "nan", int, float or None, default=None
        The value to replace NaNs and infinite values with before fitting the base
        estimator. int or float input will replace with the specified value, while
        "nan" will replace infinite values with NaNs. If None, do not replace NaNs.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

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
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of TransformerMixin
        Stores the interval extraction transformer for all estimators.
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances_ ,total_intervals * att_subsample_size)
        The transformed dataset for all estimators. Only saved when
        save_transformed_data is true.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction", Information Sciences, 239, 2013
    .. [2] Matthew Middlehurst and James Large and Anthony Bagnall. "The Canonical
       Interval Forest (CIF) Classifier for Time Series Classification."
       IEEE International Conference on Big Data 2020
    .. [3] Cabello, Nestor, et al. "Fast and Accurate Time Series Classification
       Through Supervised Interval Search." IEEE ICDM 2020

    Examples
    --------
    >>> from tsml.interval_based import IntervalForestClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=10, series_length=12, random_state=0)
    >>> clf = IntervalForestClassifier(n_estimators=10, random_state=0)
    >>> clf.fit(X, y)
    IntervalForestClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        interval_selection_method="random",
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        interval_features=None,
        series_transformers=None,
        att_subsample_size=None,
        replace_nan=None,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method=interval_selection_method,
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=series_transformers,
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
            "n_intervals": 2,
        }


class IntervalForestRegressor(RegressorMixin, BaseIntervalForest):
    """Configurable interval extracting forest regressor.

    Extracts multiple phase-dependent intervals from time series data and builds a
    base regressor on summary statistic extracted from each interval. Forms and
    ensemble of these regressors.

    Allows the implementation of regressors along the lines of [1][2][3]
    which extract intervals and create an ensemble from the subsequent features.

    By default, uses a configuration similar to TimeSeriesFroest [1].

    Parameters
    ----------
    base_estimator : BaseEstimator or None, default=None
        scikit-learn BaseEstimator used to build the interval ensemble. If None, use a
        simple decision tree.
    n_estimators : int, default=200
        Number of estimators to build for the ensemble.
    interval_selection_method : "random", default="random"
        The interval selection transformer to use.
            - "random" uses a RandomIntervalTransformer.
    n_intervals : int, str, list or tuple, default="sqrt"
        Number of intervals to extract per tree for each series_transformers series.

        An int input will extract that number of intervals from the series, while a str
        input will return a function of the series length (may differ per
        series_transformers output) to extract that number of intervals.
        Valid str inputs are:
            - "sqrt": square root of the series length.
            - "sqrt-div": sqrt of series length divided by the number
                of series_transformers.

        A list or tuple of ints and/or strs will extract the number of intervals using
        the above rules and sum the results for the final n_intervals. i.e. [4, "sqrt"]
        will extract sqrt(n_timepoints) + 4 intervals.

        Different number of intervals for each series_transformers series can be
        specified using a nested list or tuple. Any list or tuple input containing
        another list or tuple must be the same length as the number of
        series_transformers.
    min_interval_length : int, float, list, or tuple, default=3
        Minimum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the minimum interval length.

        Different minimum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.
    max_interval_length : int, float, list, or tuple, default=np.inf
        Maximum length of intervals to extract from series. float inputs take a
        proportion of the series length to use as the maximum interval length.

        Different maximum interval lengths for each series_transformers series can be
        specified using a list or tuple. Any list or tuple input must be the same length
        as the number of series_transformers.
    interval_features : TransformerMixin, callable, list, tuple, or None, default=None
        The features to extract from the intervals using transformers or callable
        functions. If None, use the mean, standard deviation, and slope of the series.

        Both transformers and functions should be able to take a 2D np.ndarray input.
        Functions should output a 1d array (the feature for each series), and
        transformers should output a 2d array where rows are the features for each
        series. A list or tuple of transformers and/or functions will extract all
        features and concatenate the output.

        Different features for each series_transformers series can be specified using a
        nested list or tuple. Any list or tuple input containing another list or tuple
        must be the same length as the number of series_transformers.
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals. If None,
        use the series as is.

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    att_subsample_size : int, float, list, tuple or None, default=None
        The number of attributes to subsample for each estimator. If None, use all

        If int, use that number of attributes for all estimators. If float, use that
        proportion of attributes for all estimators.

        Different subsample sizes for each series_transformers series can be specified
        using a list or tuple. Any list or tuple input must be the same length as the
        number of series_transformers.
    replace_nan : "nan", int, float or None, default=None
        The value to replace NaNs and infinite values with before fitting the base
        estimator. int or float input will replace with the specified value, while
        "nan" will replace infinite values with NaNs. If None, do not replace NaNs.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators are used.
    contract_max_n_estimators : int, default=500
        Max number of estimators when time_limit_in_minutes is set.
    save_transformed_data : bool, default=False
        Save the data transformed in fit.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    total_intervals_ : int
        Total number of intervals per tree from all representations.
    estimators_ : list of shape (n_estimators) of BaseEstimator
        The collections of estimators trained in fit.
    intervals_ : list of shape (n_estimators) of TransformerMixin
        Stores the interval extraction transformer for all estimators.
    transformed_data_ : list of shape (n_estimators) of ndarray with shape
    (n_instances_ ,total_intervals * att_subsample_size)
        The transformed dataset for all estimators. Only saved when
        save_transformed_data is true.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction", Information Sciences, 239, 2013
    .. [2] Matthew Middlehurst and James Large and Anthony Bagnall. "The Canonical
       Interval Forest (CIF) Classifier for Time Series Classification."
       IEEE International Conference on Big Data 2020
    .. [3] Cabello, Nestor, et al. "Fast and Accurate Time Series Classification
       Through Supervised Interval Search." IEEE ICDM 2020

    Examples
    --------
    >>> from tsml.interval_based import IntervalForestRegressor
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=10, series_length=12,
    ...                              regression_target=True, random_state=0)
    >>> reg = IntervalForestRegressor(n_estimators=10, random_state=0)
    >>> reg.fit(X, y)
    IntervalForestRegressor(...)
    >>> reg.predict(X)
    array([0.7252543 , 1.50132442, 0.95608366, 1.64399016, 0.42385504,
           0.60639322, 1.01919317, 1.30157483, 1.66017354, 0.2900776 ])
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=200,
        interval_selection_method="random",
        n_intervals="sqrt",
        min_interval_length=3,
        max_interval_length=np.inf,
        interval_features=None,
        series_transformers=None,
        att_subsample_size=None,
        replace_nan=None,
        time_limit_in_minutes=None,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            interval_selection_method=interval_selection_method,
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            max_interval_length=max_interval_length,
            interval_features=interval_features,
            series_transformers=series_transformers,
            att_subsample_size=att_subsample_size,
            replace_nan=replace_nan,
            time_limit_in_minutes=time_limit_in_minutes,
            contract_max_n_estimators=contract_max_n_estimators,
            save_transformed_data=save_transformed_data,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

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
            "n_intervals": 2,
        }
