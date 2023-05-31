# -*- coding: utf-8 -*-
"""Utilities for input validation"""

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "check_n_jobs",
    "is_transformer",
    "is_clusterer",
    "check_X_y",
    "check_X",
]

import os
import warnings
from importlib import import_module
from typing import List, Tuple, Union

import numpy as np
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    _assert_all_finite,
    _check_y,
    _ensure_no_complex_data,
    _num_samples,
    check_consistent_length,
)


def check_n_jobs(n_jobs: int) -> int:
    """Check `n_jobs` parameter according to the scikit-learn convention.

    https://scikit-learn.org/stable/glossary.html#term-n_jobs

    Parameters
    ----------
    n_jobs : int or None
        The number of jobs for parallelization.
        If None or 0, 1 is used.
        If negative, (n_cpus + 1 + n_jobs) is used. In such a case, -1 would use all
        available CPUs and -2 would use all but one.

    Returns
    -------
    n_jobs : int
        The number of threads to be used.

    Examples
    --------
    >>> from tsml.utils.validation import check_n_jobs
    >>> checked_n_jobs = check_n_jobs(-1)
    """
    if n_jobs is None or n_jobs == 0:
        return 1
    elif not isinstance(n_jobs, (int, np.integer)):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return max(1, os.cpu_count() + 1 + n_jobs)
    else:
        return n_jobs


def is_transformer(estimator: BaseEstimator) -> bool:
    """Check if an estimator is a transformer.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to check.

    Returns
    -------
    is_transformer : bool
        True if estimator is a transformer and False otherwise.
    """
    return isinstance(estimator, TransformerMixin) and isinstance(
        estimator, BaseEstimator
    )


def is_clusterer(estimator: BaseEstimator) -> bool:
    """Check if an estimator is a clusterer.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to check.

    Returns
    -------
    is_clusterer : bool
        True if estimator is a clusterer and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "clusterer"


def _num_features(X: Union[np.ndarray, List[np.ndarray]]) -> Tuple[int]:
    """Return the number of features of a 3D numpy array or a list of 2D numpy arrays.

    Returns
    -------
    num_features : tuple
        A tuple containing the number of channels, the minimum series length and the
        maximum series length of X.
    """
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return X.shape[1], X.shape[2], X.shape[2]
    elif isinstance(X, np.ndarray) and X.ndim == 2:
        return 1, X.shape[1], X.shape[1]
    elif isinstance(X, list) and isinstance(X[0], np.ndarray) and X[0].ndim == 2:
        lengths = [x.shape[1] for x in X]
        return X[0].shape[0], np.min(lengths), np.max(lengths)
    else:
        raise ValueError("X must be a 3D numpy array or a list of 2D numpy arrays")


def _check_estimator_name(estimator):
    if estimator is not None:
        if isinstance(estimator, str):
            return estimator
        else:
            return estimator.__class__.__name__
    return None


def check_X_y(
    X: object,
    y: object,
    dtype: Union[str, type, None] = "numeric",
    copy: bool = False,
    force_all_finite: bool = True,
    convert_2d: bool = True,
    ensure_min_samples: int = 1,
    ensure_min_channels: int = 1,
    ensure_min_series_length: int = 2,
    ensure_equal_length: bool = False,
    estimator: Union[str, BaseEstimator, None] = None,
    y_numeric: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 3D and y 1D. By default,
    the input is checked to be a non-empty 2D numpy array, 3D numpy array or list of
    2D numpy arrays containing only finite values. If the dtype of the array is
    object, attempt converting to float, raising on failure. Standard input checks are
    also applied to y, such as checking that y does not have np.nan or np.inf targets.

    If X input is array-like but not a 3D array or list of 2D arrays, the function
    will attempt to convert the input into a numpy array and validate it as such. 2D
    numpy arrays will be converted to a 3D numpy array of shape (n,1,m) if convert_2d
    is True.

    Uses the `scikit-learn` 1.2.1 `check_X_y` function as a base.

    Parameters
    ----------
    X : object
        Input dataset to check/convert. Ideally a 3D numpy array or a list of 2D numpy
        arrays.
    y : object
        Input labels to check/convert. Ideally a 1D numpy array..
    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (number of items for list of 2D numpy array). Setting to 0 disables this
        check.
    ensure_min_channels : int, default=1
        Make sure that the array has a minimum number of channels in its second
        axis (first axis of all items for list of 2D numpy array). Setting to 0 disables
        this check.
    ensure_min_series_length : int, default=1
        Make sure that the array has some minimum number of features/series length in
        its third axis (second axis of all items for list of 2D numpy array). Setting
        to 0 disables this check.
        The default value of 2 rejects empty datasets and non-series.
    ensure_equal_length:  bool, default=False
        Make sure that all series have the same length. Setting to False disables this.
    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.
    estimator : str, estimator instance or None, default=None
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.

    Examples
    --------
    >>> from tsml.utils.validation import check_X_y
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X, y = load_minimal_chinatown()
    >>> X, y = check_X_y(X, y, dtype=np.float32, ensure_min_series_length=8)
    """
    if y is None:
        if estimator is None:
            estimator_name = "estimator"
        else:
            estimator_name = _check_estimator_name(estimator)
        raise ValueError(
            f"{estimator_name} requires y to be passed, but the target y is None"
        )

    X = check_X(
        X,
        dtype=dtype,
        copy=copy,
        force_all_finite=force_all_finite,
        convert_2d=convert_2d,
        ensure_min_samples=ensure_min_samples,
        ensure_min_channels=ensure_min_channels,
        ensure_min_series_length=ensure_min_series_length,
        estimator=estimator,
    )

    y = _check_y(y, multi_output=False, y_numeric=y_numeric)

    check_consistent_length(X, y)

    return X, y


def check_X(
    X: object,
    dtype: Union[str, type, None] = "numeric",
    copy: bool = False,
    force_all_finite: bool = True,
    convert_2d: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_channels: int = 1,
    ensure_min_series_length: int = 2,
    ensure_equal_length: bool = False,
    estimator: Union[str, BaseEstimator, None] = None,
) -> Union[np.ndarray, list]:
    """Input validation on a numpy array or list dataset.

    By default, the input is checked to be a non-empty 2D numpy array, 3D numpy array or
    list of 2D numpy arrays containing only finite values. If the dtype of the array is
    object, attempt converting to float, raising on failure.

    If the input is array-like but not a 3D array or list of 2D arrays, the function
    will attempt to convert the input into a numpy array and validate it as such. 2D
    numpy arrays will be converted to a 3D numpy array of shape (n,1,m) if convert_2d
    is True.

    Uses the `scikit-learn` 1.2.1 `check_array` function as a base.

    Parameters
    ----------
    X : object
        Input object to check/convert. Ideally a 3D numpy array or a list of 2D numpy
        arrays.
    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
    convert_2d : bool, default=False
        Whether to convert 2D numpy arrays to 3D numpy arrays of shape (n,1,m).
    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (number of items for list of 2D numpy array). Setting to 0 disables this
        check.
    ensure_min_channels : int, default=1
        Make sure that the array has a minimum number of channels in its second
        axis (first axis of all items for list of 2D numpy array). Setting to 0 disables
        this check.
    ensure_min_series_length : int, default=1
        Make sure that the array has some minimum number of features/series length in
        its third axis (second axis of all items for list of 2D numpy array). Setting
        to 0 disables this check.
        The default value of 2 rejects empty datasets and non-series.
    estimator : str, estimator instance or None, default=None
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    Examples
    --------
    >>> from tsml.utils.validation import check_X
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X, _ = load_minimal_chinatown()
    >>> X = check_X(X, dtype=np.float32, ensure_min_series_length=8)
    """
    if isinstance(X, np.matrix):
        raise TypeError(
            "np.matrix is not supported. Please convert to a numpy array with "
            "np.asarray. For more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
        )

    if sparse.issparse(X):
        raise TypeError(
            "Sparse datatypes are currently not supported. Please convert to an "
            "accepted dense format."
        )

    # store reference to original array to check if copy is needed when
    # function returns
    X_orig = X

    # we check numpy arrays later
    if isinstance(X, np.ndarray):
        pass
    # assume this is a list of numpy arrays
    elif isinstance(X, list) and not isinstance(X[0], list):
        for i, x in enumerate(X):
            _ensure_no_complex_data(x)

            if not isinstance(x, np.ndarray):
                raise ValueError(
                    "X is a list, but does not contain only np.ndarray objects. "
                    f"Found {x} at index {i}."
                )
            if x.ndim != 2:
                raise ValueError(
                    "X is a list of np.ndarray objects, but not all arrays are 2D. "
                    f"Found {x.ndim} channels at index {i}."
                )
            if x.shape[0] != X[0].shape[0]:
                raise ValueError(
                    "X is a list of np.ndarray objects, but not all arrays have "
                    "the same number of channels. "
                    f"Found {x.shape[0]} channels at index {i} and "
                    f"{X[0].shape[0]} at index 0."
                )

        dtype_orig = [getattr(x, "dtype", None) for x in X]

        # if all arrays in the list are the same dtype, set that as the original dtype.
        # otherwise, set the original dtype to None and convert to the most prevalent
        # dtype if no conversion is already specified.
        v, c = np.unique(dtype_orig, return_counts=True)
        if len(v) == 1:
            dtype_orig = v[0]
        else:
            dtype_orig = None
            if dtype is None:
                dtype = v[np.argmax(c)]
    # attempt to convert unknown array-like objects or nested lists to numpy arrays
    elif isinstance(X, list) or hasattr(X, "__array__"):
        try:
            X = np.array(X)
            dtype_orig = getattr(X, "dtype", None)

            warnings.warn(
                "Attempted to convert array-like object to np.ndarray and succeeded. "
                "This conversion is not safe however, and we recommend input X as a 3D "
                "np.ndarray or a list of 2D np.ndarray objects.",
                stacklevel=1,
            )
        except Exception as ex:
            raise ValueError(
                "Attempted to convert array-like object to np.ndarray but failed. "
                "X must be a 3D np.ndarray or a list of 2D np.ndarray objects. "
                f"Found {type(X)}."
            ) from ex
    else:
        raise ValueError(
            "X must be a 3D np.ndarray or a list of 2D np.ndarray objects. "
            f"Found {type(X)}."
        )

    # check numpy arrays, these may have been converted from list-like objects above
    is_np = False
    if isinstance(X, np.ndarray):
        # index for series length, will be 2 if 3D and 1 if 2D
        series_idx = 2

        _ensure_no_complex_data(X)

        # convert 2D numpy arrays to univariate 3D data if enabled.
        if X.ndim == 2:
            if convert_2d:
                X = X.reshape((X.shape[0], 1, -1))
            else:
                series_idx = 1
        elif X.ndim == 1:
            raise ValueError(
                "X is a np.ndarray, but does not have 3 channels. Found 1 channel. "
                "2D arrays are automatically converted to the 3D format used by tsml. "
                "Reshape your data using X.reshape(1, -1) if it contains a single "
                "sample."
            )
        elif X.ndim != 3:
            raise ValueError(
                "X is a np.ndarray, but does not have 3 channels. "
                f"Found {X.ndim} channels.  If your data is 2D, consider "
                f"using X.reshape((X.shape[0], 1, -1)) to convert it into a univariate "
                f"format usable by tsml."
            )

        dtype_orig = getattr(X, "dtype", None)
        is_np = True

    if not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if isinstance(dtype, str) and dtype == "numeric":
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    estimator_name = _check_estimator_name(estimator)
    context = " by %s" % estimator_name if estimator is not None else ""

    if dtype is not None and dtype != dtype_orig:
        # Conversion float -> int should not contain NaN or
        # inf (numpy#14412). We cannot use casting='safe' because
        # then conversion float -> int would be disallowed.
        if X.dtype.kind == "f" and np.dtype(dtype).kind in "iu":
            _assert_all_finite(
                X,
                allow_nan=False,
            )

        if is_np:
            X = X.astype(dtype, copy=False)
            _ensure_no_complex_data(X)
        else:
            [x.astype(dtype, copy=False) for x in X]
            for x in X:
                _ensure_no_complex_data(x)

    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan". Got '
            f"{force_all_finite} instead"
        )

    if force_all_finite:
        if is_np:
            _assert_all_finite(
                X,
                allow_nan=force_all_finite == "allow-nan",
            )
        else:
            for x in X:
                _assert_all_finite(
                    x,
                    allow_nan=force_all_finite == "allow-nan",
                )

    if ensure_min_samples > 0:
        n_samples = _num_samples(X)
        if n_samples < ensure_min_samples:
            raise ValueError(
                f"Found array with {n_samples} sample(s) while a minimum of "
                f"{ensure_min_samples} is required{context}."
            )

    if ensure_min_channels > 0:
        # 2d numpy array requires more than one channel
        if is_np and series_idx == 1 and ensure_min_channels > 1:
            raise ValueError(
                f"Found 2d array with 1 channel while a minimum of "
                f"{ensure_min_channels} is required{context}."
            )
        else:
            n_channels = X.shape[1] if is_np else X[0].shape[0]

        if n_channels < ensure_min_channels:
            raise ValueError(
                f"Found array with {n_channels} channel(s) while a minimum of "
                f"{ensure_min_channels} is required{context}."
            )

    if ensure_min_series_length > 0:
        series_length = (
            X.shape[series_idx] if is_np else np.min([x.shape[1] for x in X])
        )
        if series_length < ensure_min_series_length:
            raise ValueError(
                f"Found array with {series_length} series length while a minimum of "
                f"{ensure_min_series_length} is required{context}."
            )

    if copy:
        if is_np and np.may_share_memory(X, X_orig):
            X = np.asarray(X, dtype=dtype)
        else:
            # always make a list copy but check internal arrays
            X = [
                np.asarray(x, dtype=dtype) if np.may_share_memory(x, X_orig[i]) else x
                for i, x in enumerate(X)
            ]

    return X


def _check_optional_dependency(
    package_name: str,
    package_import_name: str,
    source_name: Union[str, BaseEstimator],
):
    """Check if an optional dependency is installed and raise error if not.

    If the dependency is installed but the version is outdated, a warning is raised.

    Parameters
    ----------
    package_name : str
        Name of the package to perform an installation check for. Can include version
        requirements i.e. tsfresh or tsfresh>=0.17.0.
    package_import_name : str
        The import name of the package. i.e. for the package `scikit-learn` the import
        name is `sklearn`, while for the package `tsfresh` the import name is the same
        as the package name `tsfresh`.
    source_name : str or BaseEstimator
        Source of the check i.e. an estimator or function. If a BaseEstimator is passed
        the class name of the estimator is used.

    Raises
    ------
    ModuleNotFoundError
        Error with informative message, asking to install required the dependency.

    Examples
    --------
    >>> from tsml.utils.validation import _check_optional_dependency
    >>> _check_optional_dependency(
    ...     "scikit-learn",
    ...     "sklearn",
    ...     "_check_optional_dependency",
    ... )
    """
    if isinstance(source_name, BaseEstimator):
        source_name = source_name.__class__.__name__

    try:
        req = Requirement(package_name)
    except InvalidRequirement:
        msg_version = (
            f"Unable to find requirements from input {package_name}.Input should be a "
            f'valid package requirements str e.g. "tsfresh" or "tsfresh>=0.17.0".'
        )
        raise InvalidRequirement(msg_version)

    package_version_req = req.specifier

    try:
        # attempt to import package
        pkg_ref = import_module(package_import_name)
    except ModuleNotFoundError as e:
        # package cannot be imported
        raise ModuleNotFoundError(
            f'{source_name} has an optional dependency and requires "{package_name}" '
            f'to be installed. Run: "pip install {package_name}" or "pip install '
            f'tsml[extras]" to install all optional dependencies.'
        ) from e

    # check installed version is compatible
    if package_version_req != SpecifierSet(""):
        pkg_env_version = pkg_ref.__version__

        if pkg_env_version not in package_version_req:
            warnings.warn(
                f'{source_name} requires "{package_name}", but found version '
                f"{pkg_env_version}.",
                stacklevel=2,
            )
