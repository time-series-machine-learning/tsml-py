# -*- coding: utf-8 -*-
"""Random interval features.

A transformer for the extraction of features on randomly selected intervals.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["RandomIntervals"]

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sktime.base._base import _clone_estimator
from sktime.transformations.base import BaseTransformer
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.utils.numba.stats import (
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_quantile25,
    row_quantile75,
    row_std,
)
from sktime.utils.validation import check_n_jobs


class RandomIntervals(BaseTransformer):
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
    transformers : transformer or list of transformers, default=None,
        Deprecated for 0.16.0. Use features instead.
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

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "fit_is_empty": False,
        "capability:unequal_length": False,
    }

    def __init__(
        self,
        n_intervals=100,
        min_interval_length=3,
        max_interval_length=np.inf,
        features=None,
        transformers=None,
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

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.intervals_ = []

        self._min_interval_length = min_interval_length
        self._max_interval_length = max_interval_length
        self._features = features
        self._transform_features = []
        self._n_jobs = n_jobs

        # todo: remove this in 0.16.0
        if transformers is not None:
            warnings.warn(
                "The transformers parameter is deprecated for 0.16.0. Use the features "
                "parameter instead. The transformers parameter input will be used "
                "instead of features for this object.",
                DeprecationWarning,
            )
            self.features = transformers
            self._features = transformers
        self.transformers = transformers

        super(RandomIntervals, self).__init__()

    transformer_feature_skip = ["transform_features", "_transform_features"]

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
        self.reset()
        X, y, metadata = self._check_X_y(X=X, y=y, return_metadata=True)

        self._fit_setup(X)

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

        self.intervals_ = []
        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = None

        Xt = transformed_intervals[0]
        for i in range(1, self.n_intervals):
            Xt = np.hstack((Xt, transformed_intervals[i]))

        self._is_fitted = True

        if not hasattr(self, "_output_convert") or self._output_convert == "auto":
            X_out = self._convert_output(Xt, metadata=metadata)
        else:
            X_out = Xt
        return X_out

    def _fit(self, X, y=None):
        self._fit_setup(X)

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

        self.intervals_ = []
        for i in intervals:
            self.intervals_.extend(i)

        self._transform_features = None

        return self

    def _transform(self, X, y=None):
        if self._transform_features is None:
            transform_features = [None] * len(self.intervals_)
        else:
            count = 0
            transform_features = []
            for _ in range(self.n_intervals):
                for feature in self._features:
                    if isinstance(feature, BaseTransformer):
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

    def _fit_setup(self, X):
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.min_interval_length < 3:
            self._min_interval_length = 3

        if self.max_interval_length < self._min_interval_length:
            self._max_interval_length = self._min_interval_length
        elif self.max_interval_length > self.series_length_:
            self._max_interval_length = self.series_length_

        if self.series_length_ < 3:
            raise ValueError("Series length must be at least 3.")

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
        for f in self._features:
            if isinstance(f, BaseTransformer):
                li.append(
                    _clone_estimator(
                        f,
                        self.random_state,
                    )
                )
            elif callable(f):
                li.append(f)
            else:
                raise ValueError()
        self._features = li

        self._n_jobs = check_n_jobs(self.n_jobs)

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
            interval_start = rng.randint(
                0, self.series_length_ - self._min_interval_length
            )
            len_range = min(
                self.series_length_ - interval_start,
                self._max_interval_length,
            )
            length = (
                rng.randint(0, len_range - self._min_interval_length)
                + self._min_interval_length
                if len_range - self._min_interval_length > 0
                else self._min_interval_length
            )
            interval_end = interval_start + length
        else:
            interval_end = (
                rng.randint(0, self.series_length_ - self._min_interval_length)
                + self._min_interval_length
            )
            len_range = min(interval_end, self._max_interval_length)
            length = (
                rng.randint(0, len_range - self._min_interval_length)
                + self._min_interval_length
                if len_range - self._min_interval_length > 0
                else self._min_interval_length
            )
            interval_start = interval_end - length

        Xt = np.empty((self.n_instances_, 0)) if transform else None
        intervals = []

        for feature in self._features:
            if isinstance(feature, BaseTransformer):
                if transform:
                    t = feature.fit_transform(
                        np.expand_dims(X[:, dim, interval_start:interval_end], axis=1),
                        y,
                    )

                    if isinstance(t, pd.DataFrame):
                        t = t.to_numpy()

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
            if isinstance(feature, BaseTransformer):
                for n in self.transformer_feature_skip:
                    if hasattr(feature, n):
                        setattr(feature, n, keep_transform)
                        break
            elif not keep_transform:
                return [[0] for _ in range(X.shape[0])]

        if isinstance(feature, BaseTransformer):
            Xt = feature.transform(
                np.expand_dims(X[:, dim, interval_start:interval_end], axis=1)
            )

            if isinstance(Xt, pd.DataFrame):
                Xt = Xt.to_numpy()
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
            if isinstance(feature, BaseTransformer):
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
        params1 = {"n_intervals": 2, "features": row_mean, "max_interval_length": 10}
        params2 = {
            "n_intervals": 2,
            "features": [
                SummaryTransformer(summary_function=("min", "max"), quantiles=None),
                row_mean,
            ],
        }
        return [params1, params2]
