# -*- coding: utf-8 -*-
from functools import partial
from typing import Tuple

import numpy as np
from sklearn.utils.estimator_checks import (
    _get_check_estimator_ids,
    _maybe_mark_xfail,
    _yield_all_checks,
)

from tsml.base import BaseTimeSeriesEstimator
from tsml.tests._estimator_checks import _yield_all_time_series_checks


def parametrize_with_checks(estimators):
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


# todo
def generate_equal_multivariate_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate data for testing."""
    rng = np.random.RandomState()
    X = rng.uniform(size=(10, 2, 8))
    y = np.zeros(10)
    y[:3] = 1
    y[3:7] = 1
    rng.shuffle(y)

    return X, y
