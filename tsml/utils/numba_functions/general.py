# -*- coding: utf-8 -*-
"""General numba utilities."""

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "unique_count",
    "first_order_differences",
    "row_first_order_differences",
    "z_normalise_series",
    "z_normalise_series_2d",
    "z_normalise_series_3d",
]

from typing import Tuple

import numpy as np
from numba import njit

import tsml.utils.numba_functions.stats as stats


@njit(fastmath=True, cache=True)
def unique_count(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numba unique value count function for a 1d numpy array.

    np.unique() is supported by numba, but the return_counts parameter is not.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    unique : 1d numpy array
        The unique values in X
    counts : 1d numpy array
        The occurrence count for each unique value in X

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.general import unique_count
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> unique, counts = unique_count(X)
    """
    if X.shape[0] > 0:
        X = np.sort(X)
        unique = np.zeros(X.shape[0])
        unique[0] = X[0]
        counts = np.zeros(X.shape[0], dtype=np.int32)
        counts[0] = 1
        uc = 0

        for i in X[1:]:
            if i != unique[uc]:
                uc += 1
                unique[uc] = i
                counts[uc] = 1
            else:
                counts[uc] += 1
        return unique[: uc + 1], counts[: uc + 1]
    return np.zeros(0), np.zeros(0, dtype=np.int32)


@njit(fastmath=True, cache=True)
def first_order_differences(X: np.ndarray) -> np.ndarray:
    """Numba first order differences function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    arr : 1d numpy array of size (X.shape[0] - 1)
        The first order differences of X

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.general import first_order_differences
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> diff = first_order_differences(X)
    """
    return X[1:] - X[:-1]


@njit(fastmath=True, cache=True)
def row_first_order_differences(X: np.ndarray) -> np.ndarray:
    """Numba first order differences function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 2d numpy array of shape (X.shape[0], X.shape[1] - 1)
        The first order differences for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.general import row_first_order_differences
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> diff = row_first_order_differences(X)
    """
    return X[:, 1:] - X[:, :-1]


@njit(fastmath=True, cache=True)
def z_normalise_series(X: np.ndarray) -> np.ndarray:
    """Numba series normalization function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The normalised series

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.general import z_normalise_series
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> X_norm = z_normalise_series(X)
    """
    s = stats.std(X)
    if s > 0:
        arr = (X - stats.mean(X)) / s
    else:
        arr = X - stats.mean(X)
    return arr


@njit(fastmath=True, cache=True)
def z_normalise_series_2d(X: np.ndarray) -> np.ndarray:
    """Numba series normalization function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 2d numpy array
        The normalised series

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.general import z_normalise_series_2d
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> X_norm = z_normalise_series_2d(X)
    """
    arr = np.zeros(X.shape)
    for i in range(X.shape[0]):
        arr[i] = z_normalise_series(X[i])
    return arr


@njit(fastmath=True, cache=True)
def z_normalise_series_3d(X: np.ndarray) -> np.ndarray:
    """Numba series normalization function for a 3d numpy array.

    Parameters
    ----------
    X : 3d numpy array
        A 3d numpy array of values

    Returns
    -------
    arr : 3d numpy array
        The normalised series

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.general import z_normalise_series_3d
    >>> X = np.array([
    ...     [[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]],
    ...     [[4, 4, 4, 4, 3, 3, 3, 2, 2, 1], [8, 8, 8, 8, 7, 7, 7, 6, 6, 5]],
    ... ])
    >>> X_norm = z_normalise_series_3d(X)
    """
    arr = np.zeros(X.shape)
    for i in range(X.shape[0]):
        arr[i] = z_normalise_series_2d(X[i])
    return arr
