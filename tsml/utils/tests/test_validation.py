"""Tests for validation utilities."""

import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tsml.dummy import DummyClassifier, DummyClusterer, DummyRegressor
from tsml.utils.validation import is_clusterer


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (DummyClusterer(), True),
        (KMeans(), True),
        (DummyClassifier(), False),
        (DummyRegressor(), False),
        (StandardScaler(), False),
        (KMeans, False),
        ("drop", False),
        (None, False),
    ],
)
def test_is_clusterer(estimator, expected):
    """Test only clusterers are identified, the clusterer checks rely on this."""
    assert is_clusterer(estimator) == expected
