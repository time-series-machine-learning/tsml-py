# -*- coding: utf-8 -*-
"""Utilities for testing estimators."""

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "generate_test_estimators",
    "parametrize_with_checks",
    "generate_3d_test_data",
]

import warnings
from functools import partial
from typing import Callable, List, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import (
    _get_check_estimator_ids,
    _maybe_mark_xfail,
    _yield_all_checks,
)

import tsml.tests.estimator_checks as ts_checks
from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.discovery import all_estimators


def generate_test_estimators() -> List[BaseEstimator]:
    """Generate a list of all estimators in tsml with test parameters.

    Uses estimator parameters from `get_test_params` if available.

    If an optional dependency is not present, the estimator is skipped and a warning is
    raised.

    Returns
    -------
    estimators : list
        A list of estimators using test parameters.
    """
    classes = all_estimators()
    estimators = []
    for c in classes:
        m = getattr(c[1], "get_test_params", None)
        if callable(m):
            params = c[1].get_test_params()
        else:
            params = {}

        if isinstance(params, list):
            for p in params:
                try:
                    estimators.append(c[1](**p))
                except ModuleNotFoundError:
                    warnings.warn(
                        f"Unable to create estimator {c[0]} with parameters {p}. "
                        f"Most likely an optional dependency is not present.",
                        ImportWarning,
                    )
        else:
            try:
                estimators.append(c[1](**params))
            except ModuleNotFoundError:
                warnings.warn(
                    f"Unable to create estimator {c[0]} with parameters {params}. "
                    f"Most likely an optional dependency is not present.",
                    ImportWarning,
                )
    return estimators


def parametrize_with_checks(estimators: List[BaseEstimator]) -> Callable:
    """Pytest specific decorator for parametrizing estimator checks.

    If the estimator is a `BaseTimeSeriesEstimator` then the `tsml` checks are used,
    otherwise the `scikit-learn` checks are used.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.

    This allows to use `pytest -k` to specify which tests to run::
        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Uses the `scikit-learn` 1.2.1 `parametrize_with_checks` function as a base.

    Parameters
    ----------
    estimators : list of estimators instances
        Estimators to generated checks for.

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : Check if estimator adheres to tsml or scikit-learn conventions.

    Examples
    --------
    >>> from tsml.utils.testing import parametrize_with_checks
    >>> from tsml.interval_based import TSFRegressor
    >>> from tsml.vector import RotationForestClassifier
    >>> @parametrize_with_checks(
    ...     [TSFRegressor(), RotationForestClassifier()]
    ... )
    ... def test_tsml_compatible_estimator(estimator, check):
    ...     check(estimator)
    """
    import pytest

    def checks_generator():
        for estimator in estimators:
            checks = (
                ts_checks._yield_all_time_series_checks
                if isinstance(estimator, BaseTimeSeriesEstimator)
                else _yield_all_checks
            )
            name = type(estimator).__name__
            for check in checks(estimator):
                check = partial(check, name)
                yield _maybe_mark_xfail(estimator, check, pytest)

    return pytest.mark.parametrize(
        "estimator, check", checks_generator(), ids=_get_check_estimator_ids
    )


def generate_3d_test_data(
    n_samples: int = 10,
    n_channels: int = 1,
    series_length: int = 12,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate 3D data for testing.

    Will ensure there is at least one sample per label.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_channels : int
        The number of series channels to generate.
    series_length : int
        The number of features/series length to generate.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a float, otherwise an int.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 3D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> data, labels = generate_3d_test_data(
    ...     n_samples=20,
    ...     n_channels=2,
    ...     series_length=10,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = n_labels * rng.uniform(size=(n_samples, n_channels, series_length))
    y = X[:, 0, 0].astype(int)

    for i in range(n_labels):
        if len(y) > i:
            X[i, 0, 0] = i
            y[i] = i
    X = X * (y[:, None, None] + 1)

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    return X, y


def generate_2d_test_data(
    n_samples: int = 10,
    series_length: int = 8,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate 2D data for testing.

    Will ensure there is at least one sample per label.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    series_length : int
        The number of features/series length to generate.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a float, otherwise an int.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : np.ndarray
        Randomly generated 2D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from tsml.utils.testing import generate_2d_test_data
    >>> data, labels = generate_2d_test_data(
    ...     n_samples=20,
    ...     series_length=10,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = n_labels * rng.uniform(size=(n_samples, series_length))
    y = X[:, 0].astype(int)

    for i in range(n_labels):
        if len(y) > i:
            X[i, 0] = i
            y[i] = i
    X = X * (y[:, None] + 1)

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    return X, y


def generate_unequal_test_data(
    n_samples: int = 10,
    n_channels: int = 1,
    min_series_length: int = 6,
    max_series_length: int = 8,
    n_labels: int = 2,
    regression_target: bool = False,
    random_state: Union[int, None] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Randomly generate unequal length 3D data for testing.

    Will ensure there is at least one sample per label.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_channels : int
        The number of series channels to generate.
    min_series_length : int
        The minimum number of features/series length to generate for invidiaul series.
    max_series_length : int
        The maximum number of features/series length to generate for invidiaul series.
    n_labels : int
        The number of unique labels to generate.
    regression_target : bool
        If True, the target will be a float, otherwise an int.
    random_state : int or None
        Seed for random number generation.

    Returns
    -------
    X : list of np.ndarray
        Randomly generated unequal length 3D data.
    y : np.ndarray
        Randomly generated labels.

    Examples
    --------
    >>> from tsml.utils.testing import generate_unequal_test_data
    >>> data, labels = generate_unequal_test_data(
    ...     n_samples=20,
    ...     n_channels=2,
    ...     min_series_length=8,
    ...     max_series_length=12,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = []
    y = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        series_length = rng.randint(min_series_length, max_series_length + 1)
        x = n_labels * rng.uniform(size=(n_channels, series_length))
        label = x[0, 0].astype(int)
        if i < n_labels and n_samples > i:
            x[0, 0] = i
            label = i
        x = x * (label + 1)

        X.append(x)
        y[i] = label

    if regression_target:
        y = y.astype(np.float32)
        y += rng.uniform(size=y.shape)

    return X, y
