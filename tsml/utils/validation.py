# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]
__all__ = ["check_n_jobs"]

import os

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._array_api import get_namespace
from sklearn.utils.validation import (
    _assert_all_finite,
    _check_estimator_name,
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
    """
    if n_jobs is None or n_jobs == 0:
        return 1
    elif not isinstance(n_jobs, (int, np.integer)):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return max(1, os.cpu_count() + 1 + n_jobs)
    else:
        return n_jobs


def is_transformer(estimator):
    """Check if an estimator is a transformer. todo

    Parameters
    ----------
    estimator : object
        The estimator to check.

    Returns
    -------
    is_transformer : bool
        True if the estimator is a transformer, False otherwise.
    """
    return isinstance(estimator, TransformerMixin) and isinstance(
        estimator, BaseEstimator
    )


def is_clusterer(estimator):
    """Return True if the given estimator is (probably) a classifier.  #todo

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "clusterer"


def _num_features(X):
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return X.shape[1], X.shape[2], X.shape[2]
    elif isinstance(X, list) and isinstance(X[0], np.ndarray) and X[0].ndim == 2:
        lengths = [x.shape[1] for x in X]
        return X[0].shape[0], np.min(lengths), np.max(lengths)
    else:
        raise ValueError("X must be a 3D numpy array or a list of 2D numpy arrays")


def check_X_y(
    X,
    y,
    dtype="numeric",
    copy=False,
    force_all_finite=True,
    convert_2d=True,
    ensure_min_samples=1,
    ensure_min_dimensions=1,
    ensure_min_series_length=2,
    estimator=None,
    y_numeric=False,
):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : {ndarray, list, sparse matrix}
        Input data.

    y : {ndarray, list, sparse matrix}
        Labels.

    accept_sparse : str, bool or list of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : {'F', 'C'}, default=None
        Whether an array will be forced to be fortran or c-style.

    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in X. This parameter
        does not influence whether y can have np.inf, np.nan, pd.NA values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan or pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : bool, default=True
        Whether to raise a value error if X is not 2D.

    allow_nd : bool, default=False
        Whether to allow X.ndim > 2.

    ensure_min_samples : int, default=1
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
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
        ensure_min_dimensions=ensure_min_dimensions,
        ensure_min_series_length=ensure_min_series_length,
        estimator=estimator,
    )

    y = _check_y(y, multi_output=False, y_numeric=y_numeric, estimator=estimator)

    check_consistent_length(X, y)

    return X, y


def check_X(
    X,
    dtype="numeric",
    copy=False,
    force_all_finite=True,
    convert_2d=True,
    ensure_min_samples=1,
    ensure_min_dimensions=1,
    ensure_min_series_length=2,
    estimator=None,
):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    X : object
        Input object to check / convert.

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

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_dimensions : int, default=1
        pass

    ensure_min_series_length : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.
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

    if isinstance(X, np.ndarray):
        pass
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
                    f"Found {x.ndim} dimensions at index {i}."
                )
            if x.shape[0] != X[0].shape[0]:
                raise ValueError(
                    "X is a list of np.ndarray objects, but not all arrays have "
                    "the same number of dimensions. "
                    f"Found {x.shape[0]} dimensions at index {i} and "
                    f"{X[0].shape[0]} at index 0."
                )

            raise ValueError(
                "X must have a series length of at least 2. Found series length "
                f"{x.shape[1]} at index {i}."
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
    elif isinstance(X, list) or hasattr(X, "__array__"):
        try:
            X = np.array(X)
            dtype_orig = getattr(X, "dtype", None)
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

    is_np = False
    if isinstance(X, np.ndarray):
        _ensure_no_complex_data(X)

        if X.ndim == 2 and convert_2d:
            # convert 2D numpy arrays to univariate 3D data.
            X = X.reshape((X.shape[0], 1, -1))
        elif X.ndim == 1:
            raise ValueError(
                "X is a np.ndarray, but does not have 3 dimensions. Found 1 dimension. "
                "2D arrays are automatically converted to the 3D format used by tsml. "
                "Reshape your data using X.reshape(1, -1) if it contains a single "
                "sample."
            )
        elif X.ndim != 3:
            raise ValueError(
                "X is a np.ndarray, but does not have 3 dimensions. "
                f"Found {X.ndim} dimensions.  If your data is 2D, consider "
                f"using X.reshape((X.shape[0], 1, -1)) to convert it into a univariate "
                f"format usable by tsml."
            )

        if X.shape[2] < 2:
            raise ValueError(
                "X must have a series length of at least 2. Found series length "
                f"{X.shape[2]}."
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
                estimator_name=estimator_name,
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
        _assert_all_finite(
            X,
            allow_nan=force_all_finite == "allow-nan",
            estimator_name=estimator_name,
        )

    if ensure_min_samples > 0:
        n_samples = _num_samples(X)
        if n_samples < ensure_min_samples:
            raise ValueError(
                f"Found array with {n_samples} sample(s) while a minimum of "
                f"{ensure_min_samples} is required{context}."
            )

    if ensure_min_dimensions > 0:
        n_dimensions = X.shape[1] if is_np else X[0].shape[0]
        if n_dimensions < ensure_min_dimensions:
            raise ValueError(
                f"Found array with {n_dimensions} dimension(s) while a minimum of "
                f"{ensure_min_dimensions} is required{context}."
            )

    if ensure_min_series_length > 0:
        series_length = X.shape[2] if is_np else np.min([x.shape[1] for x in X])
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
