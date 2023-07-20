"""Interval extraction test code."""

from tsml.transformations import (
    Catch22Transformer,
    RandomIntervalTransformer,
    SevenNumberSummaryTransformer,
    SupervisedIntervalTransformer,
)
from tsml.utils.numba_functions.stats import row_mean, row_median
from tsml.utils.testing import generate_3d_test_data


def test_interval_prune():
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
    X, y = generate_3d_test_data(random_state=0, n_channels=2, series_length=10)

    rit = RandomIntervalTransformer(
        features=SevenNumberSummaryTransformer(),
        n_intervals=5,
        random_state=2,
    )
    X_t = rit.fit_transform(X, y)

    assert X_t.shape == (10, 35)
    assert rit.transform(X).shape == (10, 35)


def test_supervised_transformers():
    X, y = generate_3d_test_data(random_state=0)

    sit = SupervisedIntervalTransformer(
        features=[
            Catch22Transformer(
                features=["DN_HistogramMode_5", "SB_BinaryStats_mean_longstretch1"]
            ),
            row_mean,
        ],
        n_intervals=2,
        random_state=0,
    )
    X_t = sit.fit_transform(X, y)

    assert X_t.shape == (X.shape[0], 8)
