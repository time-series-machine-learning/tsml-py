"""Randomised Interval-Shapelet Transformation (RIST) pipeline estimators."""

__author__ = ["MatthewMiddlehurst"]

from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble._base import _set_random_states
from sklearn.utils.validation import check_is_fitted, check_random_state

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations import (
    ARCoefficientTransformer,
    Catch22Transformer,
    FunctionTransformer,
    PeriodogramTransformer,
    RandomDilatedShapeletTransformer,
    RandomIntervalTransformer,
)
from tsml.utils.numba_functions.general import first_order_differences_3d
from tsml.utils.numba_functions.stats import (
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_ppv,
    row_slope,
    row_std,
)
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class RISTClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Randomised Interval-Shapelet Transformation (RIST) pipeline classifier.

    This classifier is a hybrid pipeline using the RandomIntervalTransformer using
    Catch22 features and summary stats, and the RandomDilatedShapeletTransformer.
    Both transforms extract features from different series transformations (1st Order
    Differences, PeriodogramTransformer, and ARCoefficientTransformer).
    An ExtraTreesClassifier with 200 trees is used as the estimator for the
    concatenated feature vector output.

    Parameters
    ----------
    n_intervals : int, callable or None, default=None,
        The number of intervals of random length, position and dimension to be
        extracted for the interval portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of intervals per `series_transformer` output.
        If None, extracts `int(np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5)`
        intervals where `Xt` is the series representation data.
    n_shapelets : int, callable or None, default=None,
        The number of shapelets of random dilation and position to be extracted for the
        shapelet portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of shapelets per `series_transformer` output.
        If None, extracts `int(np.sqrt(Xt.shape[2]) * 200 + 5)` shapelets where `Xt` is
        the series representation data.
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals and
        shapelets. If None, use the series as is. If "default", use [None, 1st Order
        Differences, PeriodogramTransformer, and ARCoefficientTransformer].

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    use_pycatch22 : bool, optional, default=True
        Wraps the C based pycatch22 implementation for aeon.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
    use_pyfftw : bool, default=True
        Whether to use the pyfftw library for FFT calculations. Requires the pyfftw
        package to be installed.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to an
        ExtraTreesClassifier with 200 trees.
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
    RandomIntervalTransformer
    RandomDilatedShapeletTransformer
    RISTRegressor

    Examples
    --------
    >>> from tsml.hybrid import RISTClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> clf = RISTClassifier(random_state=0)
    >>> clf.fit(X, y)
    RISTClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 1, 0, 0, 1, 0, 1])
    """

    def __init__(
        self,
        n_intervals=None,
        n_shapelets=None,
        series_transformers="default",
        use_pycatch22=True,
        use_pyfftw=True,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_intervals = n_intervals
        self.n_shapelets = n_shapelets
        self.series_transformers = series_transformers
        self.use_pycatch22 = use_pycatch22
        self.use_pyfftw = use_pyfftw
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs

        if use_pycatch22:
            _check_optional_dependency("pycatch22", "pycatch22", self)
        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)

        super(RISTClassifier, self).__init__()

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
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=3
        )
        X = self._convert_X(X)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._estimator = _clone_estimator(
            ExtraTreesClassifier(n_estimators=200, criterion="entropy")
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t, self._series_transformers, self._transformers = _fit_transforms(
            X,
            y,
            self.series_transformers,
            self.n_intervals,
            self.n_shapelets,
            self.use_pyfftw,
            self.use_pycatch22,
            self.random_state,
            self._n_jobs,
        )
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
            Predicted class labels.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)
        X = self._convert_X(X)

        return self._estimator.predict(
            _transform_data(X, self._series_transformers, self._transformers)
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

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)
        X = self._convert_X(X)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(
                _transform_data(X, self._series_transformers, self._transformers)
            )
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(
                _transform_data(X, self._series_transformers, self._transformers)
            )
            for i in range(0, X.shape[0]):
                dists[i, self.class_dictionary_[preds[i]]] = 1
            return dists

    def _more_tags(self) -> dict:
        return {
            "optional_dependency": self.use_pycatch22 or self.use_pyfftw,
            "non_deterministic": True,
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
            "series_transformers": [
                None,
                FunctionTransformer(func=first_order_differences_3d, validate=False),
            ],
            "n_intervals": 1,
            "n_shapelets": 2,
            "estimator": ExtraTreesClassifier(n_estimators=2, criterion="entropy"),
        }


class RISTRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """Randomised Interval-Shapelet Transformation (RIST) pipeline regressor.

    This regressor is a hybrid pipeline using the RandomIntervalTransformer using
    Catch22 features and summary stats, and the RandomDilatedShapeletTransformer.
    Both transforms extract features from different series transformations (1st Order
    Differences, PeriodogramTransformer, and ARCoefficientTransformer).
    An ExtraTreesRegressor with 200 trees is used as the estimator for the
    concatenated feature vector output.

    Parameters
    ----------
    n_intervals : int, callable or None, default=None,
        The number of intervals of random length, position and dimension to be
        extracted for the interval portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of intervals per `series_transformer` output.
        If None, extracts `int(np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5)`
        intervals where `Xt` is the series representation data.
    n_shapelets : int, callable or None, default=None,
        The number of shapelets of random dilation and position to be extracted for the
        shapelet portion of the pipeline. Input should be an int or
        a function that takes a 3D np.ndarray input and returns an int. Functions may
        extract a different number of shapelets per `series_transformer` output.
        If None, extracts `int(np.sqrt(Xt.shape[2]) * 200 + 5)` shapelets where `Xt` is
        the series representation data.
    series_transformers : TransformerMixin, list, tuple, or None, default=None
        The transformers to apply to the series before extracting intervals and
        shapelets. If None, use the series as is. If "default", use [None, 1st Order
        Differences, PeriodogramTransformer, and ARCoefficientTransformer].

        A list or tuple of transformers will extract intervals from
        all transformations concatenate the output. Including None in the list or tuple
        will use the series as is for interval extraction.
    use_pycatch22 : bool, optional, default=True
        Wraps the C based pycatch22 implementation for aeon.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
    use_pyfftw : bool, default=True
        Whether to use the pyfftw library for FFT calculations. Requires the pyfftw
        package to be installed.
    estimator : sklearn classifier, default=None
        An sklearn estimator to be built using the transformed data. Defaults to an
        ExtraTreesRegressor with 200 trees.
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
    RandomIntervalTransformer
    RandomDilatedShapeletTransformer
    RISTClassifier

    Examples
    --------
    >>> from tsml.hybrid import RISTRegressor
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10,
    ...                              regression_target=True, random_state=0)
    >>> reg = RISTRegressor(random_state=0)
    >>> reg.fit(X, y)
    RISTRegressor(...)
    >>> reg.predict(X)
    array([0.31798318, 1.41426301, 1.06414747, 0.6924721 , 0.56660146,
           1.26538944, 0.52324808, 1.0939405 ])
    """

    def __init__(
        self,
        n_intervals=None,
        n_shapelets=None,
        series_transformers="default",
        use_pycatch22=True,
        use_pyfftw=True,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.n_intervals = n_intervals
        self.n_shapelets = n_shapelets
        self.series_transformers = series_transformers
        self.use_pycatch22 = use_pycatch22
        self.use_pyfftw = use_pyfftw
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs

        if use_pycatch22:
            _check_optional_dependency("pycatch22", "pycatch22", self)
        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)

        super(RISTRegressor, self).__init__()

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
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_min_series_length=3
        )
        X = self._convert_X(X)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._estimator = _clone_estimator(
            ExtraTreesRegressor(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t, self._series_transformers, self._transformers = _fit_transforms(
            X,
            y,
            self.series_transformers,
            self.n_intervals,
            self.n_shapelets,
            self.use_pyfftw,
            self.use_pycatch22,
            self.random_state,
            self._n_jobs,
        )
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

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)
        X = self._convert_X(X)

        return self._estimator.predict(
            _transform_data(X, self._series_transformers, self._transformers)
        )

    def _more_tags(self) -> dict:
        return {
            "optional_dependency": self.use_pycatch22 or self.use_pyfftw,
            "non_deterministic": True,
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
            "series_transformers": [
                None,
                FunctionTransformer(func=first_order_differences_3d, validate=False),
            ],
            "n_intervals": 1,
            "n_shapelets": 2,
            "estimator": ExtraTreesRegressor(n_estimators=2),
        }


def _fit_transforms(
    X,
    y,
    series_transformers,
    n_intervals,
    n_shapelets,
    use_pyfftw,
    use_pycatch22,
    random_state,
    n_jobs,
):
    rng = check_random_state(random_state)

    if series_transformers == "default":
        series_transformers = [
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(use_pyfftw=use_pyfftw),
            ARCoefficientTransformer(
                replace_nan=True, order=int(12 * (X.shape[2] / 100.0) ** 0.25)
            ),
        ]
    elif isinstance(series_transformers, (list, tuple)):
        series_transformers = [
            None if st is None else _clone_estimator(st, random_state=rng)
            for st in series_transformers
        ]
    else:
        series_transformers = [
            None
            if series_transformers is None
            else _clone_estimator(series_transformers, random_state=rng)
        ]

    X_t = np.empty((X.shape[0], 0))
    transformers = []
    for st in series_transformers:
        if st is not None:
            s = st.fit_transform(X, y)
        else:
            s = X

        if n_intervals is None:
            n_intervals = int(np.sqrt(X.shape[2]) * np.sqrt(X.shape[1]) * 15 + 5)
        elif callable(n_intervals):
            n_intervals = n_intervals(s)
        else:
            n_intervals = n_intervals

        ct = RandomIntervalTransformer(
            n_intervals=n_intervals,
            features=[
                Catch22Transformer(
                    outlier_norm=True, replace_nans=True, use_pycatch22=use_pycatch22
                ),
                row_mean,
                row_std,
                row_slope,
                row_median,
                row_iqr,
                row_numba_min,
                row_numba_max,
                row_ppv,
            ],
            n_jobs=n_jobs,
        )
        _set_random_states(ct, rng)
        transformers.append(ct)
        t = ct.fit_transform(s, y)

        X_t = np.hstack((X_t, t))

        if n_shapelets is None:
            n_shapelets = int(np.sqrt(X.shape[2]) * 200 + 5)
        elif callable(n_shapelets):
            n_shapelets = n_shapelets(s)
        else:
            n_shapelets = n_shapelets

        st = RandomDilatedShapeletTransformer(max_shapelets=n_shapelets, n_jobs=n_jobs)
        _set_random_states(st, rng)
        transformers.append(st)
        t = st.fit_transform(s, y)

        X_t = np.hstack((X_t, t))

    X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
    return X_t, series_transformers, transformers


def _transform_data(X, series_transformers, transformers):
    X_t = np.empty((X.shape[0], 0))
    for i, st in enumerate(series_transformers):
        if st is not None:
            s = st.transform(X)
        else:
            s = X

        t = transformers[i * 2].transform(s)
        X_t = np.hstack((X_t, t))

        t = transformers[i * 2 + 1].transform(s)
        X_t = np.hstack((X_t, t))

    X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
    return X_t
