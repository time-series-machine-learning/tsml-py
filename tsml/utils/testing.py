# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]
__all__ = []

from functools import partial
from typing import Tuple

import numpy as np
from sklearn.utils.estimator_checks import (
    _get_check_estimator_ids,
    _maybe_mark_xfail,
    _yield_all_checks,
)

from tsml.base import BaseTimeSeriesEstimator
from tsml.tests.estimator_checks import _yield_all_time_series_checks
from tsml.utils.discovery import all_estimators


def generate_test_estimators():
    classes = all_estimators()
    estimators = []
    for i, c in enumerate(classes):
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


def parametrize_with_checks(estimators):
    """scikit-learn 1.2.1 as a base"""
    import pytest

    def checks_generator():
        for estimator in estimators:
            checks = (
                _yield_all_time_series_checks
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
    n_samples=10, n_dims=1, series_length=8, random_state=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data for testing."""
    rng = np.random.RandomState(random_state)
    X = 2 * rng.uniform(size=(n_samples, n_dims, series_length))
    y = X[:, 0].astype(int)
    rng.shuffle(y)

    return X, y
