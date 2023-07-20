"""Manhattan distance."""

__author__ = ["chrisholder", "TonyBagnall", "baraline"]

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    r"""Compute the manhattan distance between two time series.

    The manhattan distance between two time series is defined as:
    .. math::
        manhattan(x, y) = \sum_{i=1}^{n} |x_i - y_i|

    Parameters
    ----------
    x: np.ndarray, of shape (n_channels, n_timepoints) or (n_timepoints)
        First time series.
    y: np.ndarray, of shape (m_channels, m_timepoints) or (m_timepoints)
        Second time series.

    Returns
    -------
    float :
        manhattan distance between x and y.

    Raises
    ------
    ValueError
        If x and y are not 1D or 2D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.distances import manhattan_distance
    >>> x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    >>> y = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    >>> manhattan_distance(x, y)
    100.0
    """
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_manhattan_distance(x, y)
    if x.ndim == 2 and y.ndim == 2:
        return _manhattan_distance(x, y)
    raise ValueError("x and y must be 1D or 2D")


@njit(cache=True, fastmath=True)
def _manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_val = min(x.shape[0], y.shape[0])
    for i in range(min_val):
        distance += _univariate_manhattan_distance(x[i], y[i])
    return distance


@njit(cache=True, fastmath=True)
def _univariate_manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    distance = 0.0
    min_length = min(x.shape[0], y.shape[0])
    for i in range(min_length):
        difference = x[i] - y[i]
        distance += abs(difference)
    return distance
