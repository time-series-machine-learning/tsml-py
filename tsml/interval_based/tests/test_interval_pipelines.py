"""Tests for the interval pipeline classes."""

from tsml.interval_based import RandomIntervalClassifier
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

    assert est._transformer._n_intervals == 2
