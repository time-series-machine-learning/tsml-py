"""Numba statistic utilities."""

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "mean",
    "row_mean",
    "count_mean_crossing",
    "row_count_mean_crossing",
    "count_above_mean",
    "row_count_above_mean",
    "quantile",
    "row_quantile",
    "quantiles",
    "row_quantiles",
    "median",
    "row_median",
    "quantile25",
    "row_quantile25",
    "quantile75",
    "row_quantile75",
    "std",
    "std2",
    "row_std",
    "numba_min",
    "row_numba_min",
    "numba_max",
    "row_numba_max",
    "slope",
    "row_slope",
    "iqr",
    "row_iqr",
    "ppv",
    "row_ppv",
    "fisher_score",
    "prime_up_to",
    "is_prime",
]

import numpy as np
from numba import njit

import tsml.utils.numba_functions.general as general_numba


@njit(fastmath=True, cache=True)
def mean(X: np.ndarray) -> float:
    """Numba mean function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    mean : float
        The mean of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import mean
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> m = mean(X)
    """
    s = 0
    for i in range(X.shape[0]):
        s += X[i]
    return s / X.shape[0]


@njit(fastmath=True, cache=True)
def row_mean(X: np.ndarray) -> np.ndarray:
    """Numba mean function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The means for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_mean
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> m = row_mean(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = mean(X[i])
    return arr


@njit(fastmath=True, cache=True)
def count_mean_crossing(X: np.ndarray) -> float:
    """Numba count above mean of first order differences for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    mean_crossing_count : float
        The count above mean of first order differences of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import count_mean_crossing
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> c = count_mean_crossing(X)
    """
    m = mean(X)
    d = general_numba.first_order_differences((X > m).astype(np.int32))
    count = 0
    for i in range(d.shape[0]):
        if d[i] != 0:
            count += 1
    return count


@njit(fastmath=True, cache=True)
def row_count_mean_crossing(X: np.ndarray) -> np.ndarray:
    """Numba count above mean of first order differences for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The count above mean of first order differences for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_count_mean_crossing
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> c = row_count_mean_crossing(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = count_mean_crossing(X[i])
    return arr


@njit(fastmath=True, cache=True)
def count_above_mean(X: np.ndarray) -> float:
    """Numba count above mean for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    mean_crossing_count : float
        The count above mean of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import count_above_mean
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> c = count_above_mean(X)
    """
    m = mean(X)
    d = X > m
    count = 0
    for i in range(d.shape[0]):
        if d[i] != 0:
            count += 1
    return count


@njit(fastmath=True, cache=True)
def row_count_above_mean(X: np.ndarray) -> np.ndarray:
    """Numba count above mean for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The count above mean for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_count_above_mean
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> c = row_count_above_mean(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = count_above_mean(X[i])
    return arr


@njit(fastmath=True, cache=True)
def quantile(X: np.ndarray, q: float) -> float:
    """Numba quantile function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values
    q : float
        The quantile to compute, must be between 0 and 1

    Returns
    -------
    quantile : float
        The quantile of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import quantile
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> q = quantile(X, 0.5)
    """
    if q < 0 or q > 1:
        raise ValueError("q must be between 0 and 1")

    idx = int(X.shape[0] * q)
    if X.shape[0] % 2 == 1:
        s = np.partition(X, idx)
        return s[idx]
    else:
        s = np.partition(X, [idx - 1, idx])
        return 0.5 * (s[idx - 1] + s[idx])


@njit(fastmath=True, cache=True)
def row_quantile(X: np.ndarray, q: float) -> np.ndarray:
    """Numba quantile function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values
    q : float
        The quantile to compute, must be between 0 and 1

    Returns
    -------
    arr : 1d numpy array
        The quantiles for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_quantile
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> q = row_quantile(X, 0.5)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = np.quantile(X[i], q)
    return arr


@njit(fastmath=True, cache=True)
def quantiles(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Numba quantile function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values
    q : 1d numpy array
        The quantiles to compute, each value must be between 0 and 1

    Returns
    -------
    arr : 1d numpy array
        The quantile of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import quantiles
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> q = quantiles(X, np.array([0.25, 0.5, 0.75]))
    """
    s = np.sort(X)
    arr = np.zeros(q.shape[0])

    for i in range(q.shape[0]):
        if q[i] < 0 or q[i] > 1:
            raise ValueError("q must be between 0 and 1")

        idx = int(X.shape[0] * q[i])
        if X.shape[0] % 2 == 1:
            arr[i] = s[idx]
        else:
            arr[i] = 0.5 * (s[idx - 1] + s[idx])

    return arr


@njit(fastmath=True, cache=True)
def row_quantiles(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Numba quantile function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values
    q : 1d numpy array
        The quantiles to compute, each value must be between 0 and 1

    Returns
    -------
    arr : 2d numpy array
        The quantiles for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_quantiles
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> q = row_quantiles(X, np.array([0.25, 0.5, 0.75]))
    """
    arr = np.zeros((X.shape[0], q.shape[0]))
    for i in range(X.shape[0]):
        arr[i] = np.quantile(X[i], q)
    return arr


@njit(fastmath=True, cache=True)
def median(X: np.ndarray) -> float:
    """Numba median function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    median : float
        The median of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import median
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> m = median(X)
    """
    return quantile(X, 0.5)


@njit(fastmath=True, cache=True)
def row_median(X: np.ndarray) -> np.ndarray:
    """Numba median function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The medians for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_median
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> m = row_median(X)
    """
    return row_quantile(X, 0.5)


@njit(fastmath=True, cache=True)
def quantile25(X: np.ndarray) -> float:
    """Numba 0.25 quantile function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    quantile : float
        The 0.25 quantile of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import quantile25
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> q = quantile25(X)
    """
    return quantile(X, 0.25)


@njit(fastmath=True, cache=True)
def row_quantile25(X: np.ndarray) -> np.ndarray:
    """Numba 0.25 quantile function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The 0.25 quantiles for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_quantile25
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> q = row_quantile25(X)
    """
    return row_quantile(X, 0.25)


@njit(fastmath=True, cache=True)
def quantile75(X: np.ndarray) -> float:
    """Numba 0.75 quantile function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    quantile : float
        The 0.75 quantile of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import quantile75
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> q = quantile75(X)
    """
    return quantile(X, 0.75)


@njit(fastmath=True, cache=True)
def row_quantile75(X: np.ndarray) -> np.ndarray:
    """Numba 0.75 quantile function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The 0.75 quantiles for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_quantile75
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> q = row_quantile75(X)
    """
    return row_quantile(X, 0.75)


@njit(fastmath=True, cache=True)
def std(X: np.ndarray) -> float:
    """Numba standard deviation function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    std : float
        The standard deviation of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import std
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> s = std(X)
    """
    m = mean(X)
    s = 0
    for i in range(X.shape[0]):
        s += (X[i] - m) ** 2
    return (s / X.shape[0]) ** 0.5


@njit(fastmath=True, cache=True)
def std2(X: np.ndarray, X_mean: float) -> float:
    """Numba standard deviation function for a 1d numpy array with pre-calculated mean.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values
    X_mean : float
        The mean of the input array

    Returns
    -------
    std : float
        The standard deviation of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import std2
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> s = std2(X, 3)
    """
    s = 0
    for i in range(X.shape[0]):
        s += (X[i] - X_mean) ** 2
    return (s / X.shape[0]) ** 0.5


@njit(fastmath=True, cache=True)
def row_std(X: np.ndarray) -> np.ndarray:
    """Numba standard deviation function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The standard deviation for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_std
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> s = row_std(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = std(X[i])
    return arr


@njit(fastmath=True, cache=True)
def numba_min(X: np.ndarray) -> float:
    """Numba min function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    min : float
        The min of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import numba_min
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> m = numba_min(X)
    """
    m = X[0]
    for i in range(1, X.shape[0]):
        if X[i] < m:
            m = X[i]
    return m


@njit(fastmath=True, cache=True)
def row_numba_min(X: np.ndarray) -> np.ndarray:
    """Numba min function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The min for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_numba_min
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> m = row_numba_min(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = numba_min(X[i])
    return arr


@njit(fastmath=True, cache=True)
def numba_max(X: np.ndarray) -> float:
    """Numba max function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    max : float
        The max of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import numba_max
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> m = numba_max(X)
    """
    m = X[0]
    for i in range(1, X.shape[0]):
        if X[i] > m:
            m = X[i]
    return m


@njit(fastmath=True, cache=True)
def row_numba_max(X: np.ndarray) -> np.ndarray:
    """Numba max function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The max for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_numba_max
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> m = row_numba_max(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = numba_max(X[i])
    return arr


@njit(fastmath=True, cache=True)
def slope(X: np.ndarray) -> float:
    """Numba slope function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    slope : float
        The slope of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import slope
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> s = slope(X)
    """
    sum_y = 0
    sum_x = 0
    sum_xx = 0
    sum_xy = 0
    for i in range(X.shape[0]):
        sum_y += X[i]
        sum_x += i
        sum_xx += i * i
        sum_xy += X[i] * i
    slope = sum_x * sum_y - X.shape[0] * sum_xy
    denom = sum_x * sum_x - X.shape[0] * sum_xx
    return 0 if denom == 0 else slope / denom


@njit(fastmath=True, cache=True)
def row_slope(X: np.ndarray) -> np.ndarray:
    """Numba slope function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The slope for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_slope
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> s = row_slope(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = slope(X[i])
    return arr


@njit(fastmath=True, cache=True)
def iqr(X: np.ndarray) -> float:
    """Numba interquartile range function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    iqr : float
        The interquartile range of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import iqr
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> i = iqr(X)
    """
    p75, p25 = np.percentile(X, [75, 25])
    return p75 - p25


@njit(fastmath=True, cache=True)
def row_iqr(X: np.ndarray) -> np.ndarray:
    """Numba interquartile range function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The interquartile range for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_iqr
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> i = row_iqr(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = iqr(X[i])
    return arr


@njit(fastmath=True, cache=True)
def ppv(X: np.ndarray) -> float:
    """Numba proportion of positive values function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    ppv : float
        The proportion of positive values range of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import ppv
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> p = ppv(X)
    """
    count = 0
    for i in range(X.shape[0]):
        if X[i] > 0:
            count += 1
    return count / X.shape[0]


@njit(fastmath=True, cache=True)
def row_ppv(X: np.ndarray) -> np.ndarray:
    """Numba proportion of positive values function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The proportion of positive values for axis 0 of the input array

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import row_ppv
    >>> X = np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]])
    >>> p = row_ppv(X)
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = ppv(X[i])
    return arr


@njit(fastmath=True, cache=True)
def fisher_score(X: np.ndarray, y: np.ndarray) -> float:
    """Numba Fisher score function.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of attribute values
    y : 1d numpy array
        A 1d numpy array of class values

    Returns
    -------
    score : float
        The Fisher score for the given array of attribute values and class values

    Examples
    --------
    >>> import numpy as np
    >>> from tsml.utils.numba_functions.stats import fisher_score
    >>> X = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    >>> y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>> f = fisher_score(X, y)
    """
    unique_labels = np.unique(y)
    mu_feat = mean(X)
    accum_numerator = 0
    accum_denominator = 0

    for k in unique_labels:
        idx_label = np.where(y == k)[0]
        data_sub = X[idx_label]

        mu_feat_label = mean(data_sub)
        sigma_feat_label = max(std2(data_sub, mu_feat_label), 0.000001)

        accum_numerator += idx_label.shape[0] * (mu_feat_label - mu_feat) ** 2
        accum_denominator += idx_label.shape[0] * sigma_feat_label**2

    if accum_denominator == 0:
        return 0
    else:
        return accum_numerator / accum_denominator


@njit(fastmath=True, cache=True)
def prime_up_to(n: int) -> np.ndarray:
    """Check if any number from 1 to n is a prime number and return the ones which are.

    Parameters
    ----------
    n : int
        Number up to which the search for prime number will go

    Returns
    -------
    array
        Prime numbers up to n

    Examples
    --------
    >>> from tsml.utils.numba_functions.stats import prime_up_to
    >>> p = prime_up_to(50)
    """
    is_p = np.zeros(n + 1, dtype=np.bool_)
    for i in range(n + 1):
        is_p[i] = is_prime(i)
    return np.where(is_p)[0]


@njit(fastmath=True, cache=True)
def is_prime(n: int) -> bool:
    """Check if the input number is a prime number.

    Parameters
    ----------
    n : int
        The number to test

    Returns
    -------
    bool
        Wheter n is a prime number

    Examples
    --------
    >>> from tsml.utils.numba_functions.stats import is_prime
    >>> p = is_prime(7)
    """
    if (n % 2 == 0 and n > 2) or n == 0 or n == 1:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if not n % i:
            return False
    return True
