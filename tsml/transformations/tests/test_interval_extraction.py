"""Interval extraction test code."""

import pytest

from tsml.transformations import (
    RandomIntervalTransformer,
    SevenNumberSummaryTransformer,
    SupervisedIntervalTransformer,
)
from tsml.utils.numba_functions.stats import row_mean, row_median
from tsml.utils.testing import generate_3d_test_data
from tsml.utils.validation import _check_optional_dependency


def test_interval_prune():
    """Test RandomIntervalTransformer duplicate pruning."""
    X, y = generate_3d_test_data(random_state=0, n_channels=2, series_length=10)

    rit = RandomIntervalTransformer(
        features=[row_mean, row_median],
        n_intervals=10,
        random_state=0,
    )
    X_t = rit.fit_transform(X, y)

    assert X_t.shape == (10, 16)
    assert rit.transform(X).shape == (10, 16)


def test_random_interval_transformer():
    """Test RandomIntervalTransformer."""
    X, y = generate_3d_test_data(random_state=0, n_channels=2, series_length=10)

    rit = RandomIntervalTransformer(
        features=SevenNumberSummaryTransformer(),
        n_intervals=5,
        random_state=2,
    )
    X_t = rit.fit_transform(X, y)

    assert X_t.shape == (10, 35)
    assert rit.transform(X).shape == (10, 35)


@pytest.mark.skipif(
    not _check_optional_dependency("pycatch22", "pycatch22", None, raise_error=False),
    reason="pycatch22 not installed",
)
def test_supervised_transformers():
    """Test SupervisedIntervalTransformer."""
    X, y = generate_3d_test_data(random_state=0)

    sit = SupervisedIntervalTransformer(
        features=[
            SevenNumberSummaryTransformer(),
            row_mean,
        ],
        n_intervals=2,
        random_state=0,
    )
    X_t = sit.fit_transform(X, y)

    assert X_t.shape == (X.shape[0], 8)
