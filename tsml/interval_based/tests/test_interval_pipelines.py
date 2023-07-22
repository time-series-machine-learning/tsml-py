"""Tests for the interval pipeline classes."""

from tsml.interval_based import RandomIntervalClassifier
from tsml.transformations import FunctionTransformer
from tsml.utils.numba_functions.general import first_order_differences_3d
from tsml.utils.testing import generate_3d_test_data


def test_random_interval_callable():
    """Test RandomIntervalClassifier with a callable n_intervals."""
    X, y = generate_3d_test_data()

    def interval_func(X):
        return int(X.shape[2] / 5)

    est = RandomIntervalClassifier(
        n_intervals=interval_func,
    )
    est.fit(X, y)

    assert est._transformers[0]._n_intervals == 2


def test_random_interval_series_transform_callable():
    """Test RandomIntervalClassifier with a series transformer."""
    X, y = generate_3d_test_data()

    est = RandomIntervalClassifier(
        n_intervals=2,
        series_transformers=[
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
        ],
    )
    est.fit(X, y)
    est.predict_proba(X)

    assert len(est._transformers) == 2
