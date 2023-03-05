# -*- coding: utf-8 -*-
"""Utilities for testing estimators."""

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "generate_test_estimators",
    "parametrize_with_checks",
    "generate_test_data",
]

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
                estimators.append(c[1](**p))
        else:
            estimators.append(c[1](**params))
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
    ... def test_sklearn_compatible_estimator(estimator, check):
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


def generate_test_data(
    n_samples: int = 10,
    n_dims: int = 1,
    series_length: int = 8,
    n_labels: int = 2,
    random_state: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate 3D data for testing.

    Will ensure there is at least one sample per label.

    Parameters
    ----------
    n_samples : int
        The number of samples to generate.
    n_dims : int
        The number of series dimensions to generate.
    series_length : int
        The number of features/series length to generate.
    n_labels : int
        The number of unique labels to generate.
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
    >>> from tsml.utils.testing import generate_test_data
    >>> data, labels = generate_test_data(
    ...     n_samples=20,
    ...     n_dims=2,
    ...     series_length=10,
    ...     n_labels=3,
    ... )
    """
    rng = np.random.RandomState(random_state)
    X = n_labels * rng.uniform(size=(n_samples, n_dims, series_length))
    y = X[:, 0, 0].astype(int)
    for i in range(n_labels):
        if len(y) > i:
            X[i, 0, 0] = i
            y[i] = i
    X = X * (y[:, None, None] + 1)
    return X, y
