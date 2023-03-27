# -*- coding: utf-8 -*-
"""Random interval features.

A transformer for the extraction of features on randomly selected intervals.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["RandomIntervalTransformer", "SupervisedIntervalTransformer"]

import numpy as np
from joblib import Parallel
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.utils.numba_functions.general import z_normalise_series_3d
from tsml.utils.numba_functions.stats import (
    fisher_score,
    row_count_above_mean,
    row_count_mean_crossing,
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_quantile25,
    row_quantile75,
    row_slope,
    row_std,
)
from tsml.utils.validation import check_n_jobs, is_transformer


class RandomIntervalTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Random interval feature transformer.

    Extracts intervals with random length, position and dimension from series in fit.
    Transforms each interval sub-series using the given transformer(s)/features and
    concatenates them into a feature vector in transform.

    Parameters
    ----------
    n_intervals : int, default=100,
        The number of intervals of random length, position and dimension to be
        extracted.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    max_interval_length : int, default=3
        The maximum length of extracted intervals. Minimum value of min_interval_length.
    features : sktime transformer, a function taking a 2d numpy array parameter, or list
            of said transformers and functions, default=None
        Transformers and functions used to extract features from selected intervals.
        If None, defaults to [mean, median, min, max, std, 25% quantile, 75% quantile]
    random_state : int or None, default=None
        Seed for random number generation.
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
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    intervals_ : list of tuples
        Contains information for each feature extracted in fit. Each tuple contains the
        interval start, interval end, interval dimension and the feature(s) extracted.
        Length will be n_intervals*len(features).

    See Also
    --------
    SupervisedIntervals
    """

    def __init__(
        self,
        n_intervals=100,
        min_interval_length=3,
        max_interval_length=np.inf,
        features=None,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.features = features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super(RandomIntervalTransformer, self).__init__()

    transformer_feature_skip = ["transform_features_", "_transform_features"]

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        _X : X, coerced copy of X, if remember_data tag is True
            possibly coerced to inner type or update_data compatible type
            by reference, when possible
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |   `X`    | `tf-output`  |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        Explicitly, with examples:
            if `X` is `Series` (e.g., `pd.DataFrame`) and `transform-output` is `Series`
                then the return is a single `Series` of the same mtype
                Example: detrending a single series
            if `X` is `Panel` (e.g., `pd-multiindex`) and `transform-output` is `Series`
                then the return is `Panel` with same number of instances as `X`
                    (the transformer is applied to each input Series instance)
                Example: all series in the panel are detrended individually
            if `X` is `Series` or `Panel` and `transform-output` is `Primitives`
                then the return is `pd.DataFrame` with as many rows as instances in `X`
                Example: i-th row of the return has mean and variance of the i-th series
            if `X` is `Series` and `transform-output` is `Panel`
                then the return is a `Panel` object of type `pd-multiindex`
                Example: i-th instance of the output is the i-th window running over `X`
        """
        X, y = self._fit_setup(X, y)

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_interval)(
                X,
                y,
                i,
                True,
            )
            for i in range(self.n_intervals)
        )

        (
            intervals,
            transformed_intervals,
        ) = zip(*fit)

        for i in intervals:
            self.intervals_.extend(i)

        Xt = transformed_intervals[0]
        for i in range(1, self.n_intervals):
            Xt = np.hstack((Xt, transformed_intervals[i]))

        return Xt

    def fit(self, X, y=None):
        X, y = self._fit_setup(X, y)

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_interval)(
                X,
                y,
                i,
                False,
            )
            for i in range(self.n_intervals)
        )

        (
            intervals,
            _,
        ) = zip(*fit)

        for i in intervals:
            self.intervals_.extend(i)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)

        if self._transform_features is None:
            transform_features = [None] * len(self.intervals_)
        else:
            count = 0
            transform_features = []
            for _ in range(self.n_intervals):
                for feature in self._features:
                    if is_transformer(feature):
                        nf = feature.n_transformed_features
                        transform_features.append(
                            self._transform_features[count : count + nf]
                        )
                        count += nf
                    else:
                        transform_features.append(self._transform_features[count])
                        count += 1

        transform = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._transform_interval)(
                X,
                i,
                transform_features[i],
            )
            for i in range(len(self.intervals_))
        )

        Xt = transform[0]
        for i in range(1, len(self.intervals_)):
            Xt = np.hstack((Xt, transform[i]))

        return Xt

    def _fit_setup(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_series_length=3)

        self.intervals_ = []
        self._transform_features = None

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        self._min_interval_length = self.min_interval_length
        if self.min_interval_length < 3:
            self._min_interval_length = 3

        self._max_interval_length = self.max_interval_length
        if self.max_interval_length < self._min_interval_length:
            self._max_interval_length = self._min_interval_length
        elif self.max_interval_length > self.series_length_:
            self._max_interval_length = self.series_length_

        self._features = self.features
        if self.features is None:
            self._features = [
                row_mean,
                row_std,
                row_numba_min,
                row_numba_max,
                row_median,
                row_quantile25,
                row_quantile75,
            ]
        elif not isinstance(self.features, list):
            self._features = [self.features]

        li = []
        for feature in self._features:
            if is_transformer(feature):
                li.append(
                    _clone_estimator(
                        feature,
                        self.random_state,
                    )
                )
            elif callable(feature):
                li.append(feature)
            else:
                raise ValueError()  # todo
        self._features = li

        self._n_jobs = check_n_jobs(self.n_jobs)

        return X, y

    def _generate_interval(self, X, y, idx, transform):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        dim = rng.randint(self.n_dims_)

        if rng.random() < 0.5:
            interval_start = (
                rng.randint(0, self.series_length_ - self._min_interval_length)
                if self.series_length_ > self._min_interval_length
                else 0
            )
            len_range = min(
                self.series_length_ - interval_start,
                self._max_interval_length,
            )
            length = (
                rng.randint(0, len_range - self._min_interval_length)
                + self._min_interval_length
                if len_range > self._min_interval_length
                else self._min_interval_length
            )
            interval_end = interval_start + length
        else:
            interval_end = (
                rng.randint(0, self.series_length_ - self._min_interval_length)
                + self._min_interval_length
                if self.series_length_ > self._min_interval_length
                else self._min_interval_length
            )
            len_range = min(interval_end, self._max_interval_length)
            length = (
                rng.randint(0, len_range - self._min_interval_length)
                + self._min_interval_length
                if len_range > self._min_interval_length
                else self._min_interval_length
            )
            interval_start = interval_end - length

        Xt = np.empty((self.n_instances_, 0)) if transform else None
        intervals = []

        for feature in self._features:
            if is_transformer(feature):
                if transform:
                    feature = _clone_estimator(
                        feature,
                        rs,
                    )

                    t = feature.fit_transform(
                        np.expand_dims(X[:, dim, interval_start:interval_end], axis=1),
                        y,
                    )

                    Xt = np.hstack((Xt, t))
                else:
                    feature.fit(
                        np.expand_dims(X[:, dim, interval_start:interval_end], axis=1),
                        y,
                    )
            elif transform:
                t = [[f] for f in feature(X[:, dim, interval_start:interval_end])]
                Xt = np.hstack((Xt, t))

            intervals.append((interval_start, interval_end, dim, feature))

        return intervals, Xt

    def _transform_interval(self, X, idx, keep_transform):
        interval_start, interval_end, dim, feature = self.intervals_[idx]

        if keep_transform is not None:
            if is_transformer(feature):
                for n in self.transformer_feature_skip:
                    if hasattr(feature, n):
                        setattr(feature, n, keep_transform)
                        break
            elif not keep_transform:
                return [[0] for _ in range(X.shape[0])]

        if is_transformer(feature):
            Xt = feature.transform(
                np.expand_dims(X[:, dim, interval_start:interval_end], axis=1)
            )

        else:
            Xt = [[f] for f in feature(X[:, dim, interval_start:interval_end])]

        return Xt

    def set_features_to_transform(self, arr, raise_error=True):
        """Set transform_features to the given array.

        Each index in the list corresponds to the index of an interval, True intervals
        are included in the transform, False intervals skipped and are set to 0.

        If any transformers are in features, they must also have a "transform_features"
        or "_transform_features" attribute as well as a "n_transformed_features"
        attribute. The input array should contain an item for each of the transformers
        "n_transformed_features" output features.

        Parameters
        ----------
        arr : list of bools
             A list of intervals to skip.
        raise_error : bool, default=True
             Whether to raise and error or return None if input or transformers are
             invalid.

        Returns
        -------
        completed: bool
            Whether the operation was successful.
        """
        length = 0
        for feature in self._features:
            if is_transformer(feature):
                if not any(
                    hasattr(feature, n) for n in self.transformer_feature_skip
                ) or not hasattr(feature, "n_transformed_features"):
                    if raise_error:
                        raise ValueError(
                            "Transformer must have one of {} as an attribute and a "
                            "n_transformed_features attribute.".format(
                                self.transformer_feature_skip
                            )
                        )
                    else:
                        return False

                length += feature.n_transformed_features
            else:
                length += 1

        if len(arr) != length * self.n_intervals or not all(
            isinstance(b, bool) for b in arr
        ):
            if raise_error:
                raise ValueError(
                    "Input must be a list bools, matching the length of the transform "
                    "output."
                )
            else:
                return False

        self._transform_features = arr

        return True

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"n_intervals": 2}


class SupervisedIntervalTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Supervised interval feature transformer.

    Extracts intervals in fit using the supervised process described in [1].
    Interval sub-series are extracted for each input feature, and the usefulness of that
    feature extracted on an interval is evaluated using the Fisher score metric.
    Intervals are continually split in half, with the better scoring half retained as a
    feature for the transform.

    Multivariate capability is added by running the supervised interval extraction
    process on each dimension of the input data.

    As the extracted interval features are already extracted for the supervised
    evaluation in fit, the fit_transform method is recommended if the transformed fit
    data is required.

    Parameters
    ----------
    n_intervals : int, default=50
        The number of times the supervised interval selection process is run.
        Each supervised extraction will output a varying amount of features based on
        series length, number of dimensions and the number of features.
    min_interval_length : int, default=3
        The minimum length of extracted intervals. Minimum value of 3.
    features : function with a single 2d array-like parameter or list of said functions,
            default=None
        Functions used to extract features from selected intervals. If None, defaults to
        the following statistics used in [2]:
        [mean, median, std, slope, min, max, iqr, count_mean_crossing,
        count_above_mean].
    randomised_split_point : bool, default=True
        If True, the split point for interval extraction is randomised as is done in [2]
        rather than split in half.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    intervals_ : list of tuples
        Contains information for each feature extracted in fit. Each tuple contains the
        interval start, interval end, interval dimension and the feature extracted.
        Length will be the same as the amount of transformed features.

    See Also
    --------
    RandomIntervals

    Notes
    -----
    Based on the authors (stevcabello) code: https://github.com/stevcabello/r-STSF/

    References
    ----------
    .. [1] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2020, November. Fast and
        accurate time series classification through supervised interval search. In 2020
        IEEE International Conference on Data Mining (ICDM) (pp. 948-953). IEEE.
    .. [2] Cabello, N., Naghizade, E., Qi, J. and Kulik, L., 2021. Fast, accurate and
        interpretable time series classification through randomization. arXiv preprint
        arXiv:2105.14876.

    Examples
    """

    def __init__(
        self,
        n_intervals=50,
        min_interval_length=3,
        features=None,
        randomised_split_point=True,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.features = features
        self.randomised_split_point = randomised_split_point
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        super(SupervisedIntervalTransformer, self).__init__()

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits the transformer to X and y and returns a transformed version of X.

        State change:
            Changes state to "fitted".

        Writes to self:
        _is_fitted : flag is set to True.
        _X : X, coerced copy of X, if remember_data tag is True
            possibly coerced to inner type or update_data compatible type
            by reference, when possible
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to be transformed, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        type depends on type of X and scitype:transform-output tag:
            |   `X`    | `tf-output`  |     type of return     |
            |----------|--------------|------------------------|
            | `Series` | `Primitives` | `pd.DataFrame` (1-row) |
            | `Panel`  | `Primitives` | `pd.DataFrame`         |
            | `Series` | `Series`     | `Series`               |
            | `Panel`  | `Series`     | `Panel`                |
            | `Series` | `Panel`      | `Panel`                |
        instances in return correspond to instances in `X`
        combinations not in the table are currently not supported

        Explicitly, with examples:
            if `X` is `Series` (e.g., `pd.DataFrame`) and `transform-output` is `Series`
                then the return is a single `Series` of the same mtype
                Example: detrending a single series
            if `X` is `Panel` (e.g., `pd-multiindex`) and `transform-output` is `Series`
                then the return is `Panel` with same number of instances as `X`
                    (the transformer is applied to each input Series instance)
                Example: all series in the panel are detrended individually
            if `X` is `Series` or `Panel` and `transform-output` is `Primitives`
                then the return is `pd.DataFrame` with as many rows as instances in `X`
                Example: i-th row of the return has mean and variance of the i-th series
            if `X` is `Series` and `transform-output` is `Panel`
                then the return is a `Panel` object of type `pd-multiindex`
                Example: i-th instance of the output is the i-th window running over `X`
        """
        X, y = self._fit_setup(X, y)

        X_norm = z_normalise_series_3d(X)

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X,
                X_norm,
                y,
                i,
                True,
            )
            for i in range(self.n_intervals)
        )

        (
            intervals,
            transformed_intervals,
        ) = zip(*fit)

        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = [True] * len(self.intervals_)

        Xt = transformed_intervals[0]
        for i in range(1, self.n_intervals):
            Xt = np.hstack((Xt, transformed_intervals[i]))

        return Xt

    def fit(self, X, y=None):
        X, y = self._fit_setup(X, y)

        X_norm = z_normalise_series_3d(X)

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X,
                X_norm,
                y,
                i,
                False,
            )
            for i in range(self.n_intervals)
        )

        (
            intervals,
            _,
        ) = zip(*fit)

        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = [True] * len(self.intervals_)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=7)

        transform = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._transform_intervals)(
                X,
                i,
            )
            for i in range(len(self.intervals_))
        )

        Xt = np.zeros((X.shape[0], len(transform)))
        for i, t in enumerate(transform):
            Xt[:, i] = t

        return Xt

    def _fit_setup(self, X, y):
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=7
        )

        self.intervals_ = []

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.n_instances_ <= 1:
            raise ValueError(
                "Supervised intervals requires more than 1 training time series."
            )

        self._min_interval_length = self.min_interval_length
        if self.min_interval_length < 3:
            self._min_interval_length = 3

        if self._min_interval_length * 2 + 1 > self.series_length_:
            raise ValueError(
                "Minimum interval length must be less than half the series length."
            )

        self._features = self.features
        if self.features is None:
            self._features = [
                row_mean,
                row_median,
                row_std,
                row_slope,
                row_numba_min,
                row_numba_max,
                row_iqr,
                row_count_mean_crossing,
                row_count_above_mean,
            ]

        li = []
        if not isinstance(self._features, list):
            self._features = [self._features]

        for f in self._features:
            if callable(f):
                li.append(f)
            else:
                raise ValueError()
        self._features = li

        self._n_jobs = check_n_jobs(self.n_jobs)

        le = preprocessing.LabelEncoder()
        return X, le.fit_transform(y)

    def _generate_intervals(self, X, X_norm, y, idx, keep_transform):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        Xt = np.empty((self.n_instances_, 0)) if keep_transform else None
        intervals = []

        for i in range(self.n_dims_):
            for feature in self._features:
                random_cut_point = int(rng.randint(1, self.series_length_ - 1))
                while (
                    self.series_length_ - random_cut_point
                    < self._min_interval_length * 2
                    and self.series_length_ - (self.series_length_ - random_cut_point)
                    < self._min_interval_length * 2
                ):
                    random_cut_point = int(rng.randint(1, self.series_length_ - 1))

                intervals_L, Xt_L = self._supervised_search(
                    X_norm[:, i, :random_cut_point],
                    y,
                    0,
                    feature,
                    i,
                    X[:, i, :],
                    rng,
                    keep_transform,
                )
                intervals.extend(intervals_L)

                if keep_transform:
                    Xt = np.hstack((Xt, Xt_L))

                intervals_R, Xt_R = self._supervised_search(
                    X_norm[:, i, random_cut_point:],
                    y,
                    random_cut_point,
                    feature,
                    i,
                    X[:, i, :],
                    rng,
                    keep_transform,
                )
                intervals.extend(intervals_R)

                if keep_transform:
                    Xt = np.hstack((Xt, Xt_R))

        return intervals, Xt

    def _transform_intervals(self, X, idx):
        if not self._transform_features[idx]:
            return np.zeros(X.shape[0])

        start, end, dim, feature = self.intervals_[idx]
        return feature(X[:, dim, start:end])

    def _supervised_search(
        self, X, y, ini_idx, feature, dim, X_ori, rng, keep_transform
    ):
        intervals = []
        Xt = np.empty((X.shape[0], 0)) if keep_transform else None

        while X.shape[1] >= self._min_interval_length * 2:
            if (
                self.randomised_split_point
                and X.shape[1] != self._min_interval_length * 2
            ):
                div_point = rng.randint(
                    self._min_interval_length, X.shape[1] - self._min_interval_length
                )
            else:
                div_point = int(X.shape[1] / 2)

            sub_interval_0 = X[:, :div_point]
            sub_interval_1 = X[:, div_point:]

            interval_feature_0 = feature(sub_interval_0)
            interval_feature_1 = feature(sub_interval_1)

            score_0 = fisher_score(interval_feature_0, y)
            score_1 = fisher_score(interval_feature_1, y)

            if score_0 >= score_1 and score_0 != 0:
                end = ini_idx + len(sub_interval_0[0])

                intervals.append((ini_idx, end, dim, feature))
                X = sub_interval_0

                interval_feature_to_use = feature(X_ori[:, ini_idx:end])
                if keep_transform:
                    Xt = np.hstack(
                        (
                            Xt,
                            np.reshape(
                                interval_feature_to_use,
                                (interval_feature_to_use.shape[0], 1),
                            ),
                        )
                    )
            elif score_1 > score_0:
                ini_idx = ini_idx + div_point
                end = ini_idx + len(sub_interval_1[0])

                intervals.append((ini_idx, end, dim, feature))
                X = sub_interval_1

                interval_feature_to_use = feature(X_ori[:, ini_idx:end])
                if keep_transform:
                    Xt = np.hstack(
                        (
                            Xt,
                            np.reshape(
                                interval_feature_to_use,
                                (interval_feature_to_use.shape[0], 1),
                            ),
                        )
                    )
            else:
                break

        return intervals, Xt

    def set_features_to_transform(self, arr, raise_error=True):
        """Set transform_features to the given array.

        Each index in the list corresponds to the index of an interval, True intervals
        are included in the transform, False intervals skipped and are set to 0.

        Parameters
        ----------
        arr : list of bools
             A list of intervals to skip.
        raise_error : bool, default=True
             Whether to raise and error or return None if input is invalid.

        Returns
        -------
        completed: bool
            Whether the operation was successful.
        """
        if len(arr) != len(self.intervals_) or not all(
            isinstance(b, bool) for b in arr
        ):
            if raise_error:
                raise ValueError(
                    "Input must be a list bools of length len(intervals_)."
                )
            else:
                return False

        self._transform_features = arr

        return True

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {
            "n_intervals": 1,
            "randomised_split_point": False,
        }
