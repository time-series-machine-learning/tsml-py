"""Tests for the channel ensemble estimators."""

import pytest
from numpy.testing import assert_array_equal
from sklearn.mixture import GaussianMixture

from tsml.compose._channel_ensemble import (
    ChannelEnsembleClassifier,
    ChannelEnsembleRegressor,
    _check_key_type,
    _get_channel,
)
from tsml.dummy import DummyClassifier, DummyRegressor
from tsml.utils.testing import generate_3d_test_data, generate_unequal_test_data


def test_single_estimator():
    """Test that a single estimator is correctly applied to all channels."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(estimators=[("d", DummyClassifier(), "all")])
    ens.fit(X, y)

    assert len(ens.estimators_[0][2]) == 3
    assert ens.predict(X).shape == (X.shape[0],)

    ens = ChannelEnsembleRegressor(estimators=[("d", DummyRegressor(), "all")])
    ens.fit(X, y)

    assert len(ens.estimators_[0][2]) == 3
    assert ens.predict(X).shape == (X.shape[0],)


def test_single_estimator_split():
    """Test that a single split estimator correctly creates an estimator per channel."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(estimators=("d", DummyClassifier(), "all-split"))
    ens.fit(X, y)

    assert len(ens.estimators_) == 3
    assert isinstance(ens.estimators_[0][2], int)
    assert ens.predict(X).shape == (X.shape[0],)

    ens = ChannelEnsembleRegressor(estimators=("d", DummyRegressor(), "all-split"))
    ens.fit(X, y)

    assert len(ens.estimators_) == 3
    assert isinstance(ens.estimators_[0][2], int)
    assert ens.predict(X).shape == (X.shape[0],)


def test_remainder():
    """Test that the remainder is applied to remaining channels."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(
        estimators=[("d", DummyClassifier(), 0)],
        remainder=DummyClassifier(),
    )
    ens.fit(X, y)

    assert len(ens._remainder[2]) == 2
    assert ens.predict(X).shape == (X.shape[0],)

    ens = ChannelEnsembleRegressor(
        estimators=[("d", DummyRegressor(), 0)],
        remainder=DummyRegressor(),
    )
    ens.fit(X, y)

    assert len(ens._remainder[2]) == 2
    assert ens.predict(X).shape == (X.shape[0],)


def test_invalid_estimator_type():
    """Test that estimators of the wrong type are rejected."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleRegressor(estimators=[("d", DummyClassifier(), "all")])
    with pytest.raises(TypeError, match="correct estimator type"):
        ens.fit(X, y)

    # has predict_proba, but is not a classifier
    ens = ChannelEnsembleClassifier(estimators=[("d", GaussianMixture(), "all")])
    with pytest.raises(TypeError, match="correct estimator type"):
        ens.fit(X, y)

    ens = ChannelEnsembleClassifier(estimators=[("d", DummyRegressor(), "all")])
    with pytest.raises(TypeError, match="correct estimator type"):
        ens.fit(X, y)

    ens = ChannelEnsembleClassifier(estimators=[("d", "not an estimator", "all")])
    with pytest.raises(TypeError, match="correct estimator type"):
        ens.fit(X, y)


def test_invalid_remainder():
    """Test that a remainder of the wrong type is rejected."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(
        estimators=[("d", DummyClassifier(), 0)],
        remainder=DummyRegressor(),
    )
    with pytest.raises(ValueError, match="remainder"):
        ens.fit(X, y)

    ens = ChannelEnsembleRegressor(
        estimators=[("d", DummyRegressor(), 0)],
        remainder=DummyClassifier(),
    )
    with pytest.raises(ValueError, match="remainder"):
        ens.fit(X, y)


@pytest.mark.parametrize(
    "data_func", [generate_3d_test_data, generate_unequal_test_data]
)
@pytest.mark.parametrize("key", [[1], [0, 2]])
def test_channel_selection(data_func, key):
    """Test that channel selection works correctly."""
    X, _ = data_func(n_channels=3)

    assert _check_key_type(key)

    channels = _get_channel(X, key)

    assert channels[0].shape[0] == len(key)
    for i, k in enumerate(key):
        assert_array_equal(channels[0][i, :], X[0][k, :])
