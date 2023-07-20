"""Tests for the channel ensemble estimators."""
import pytest
from numpy.testing import assert_array_equal

from tsml.compose._channel_ensemble import (
    ChannelEnsembleClassifier,
    ChannelEnsembleRegressor,
    _check_key_type,
    _get_channel,
)
from tsml.interval_based import TSFClassifier, TSFRegressor
from tsml.utils.testing import generate_3d_test_data, generate_unequal_test_data


def test_single_estimator():
    """Test that a single estimator is correctly applied to all channels."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(
        estimators=[("tsf", TSFClassifier(n_estimators=2), "all")]
    )
    ens.fit(X, y)

    assert len(ens.estimators_[0][2]) == 3
    assert ens.predict(X).shape == (X.shape[0],)

    ens = ChannelEnsembleRegressor(
        estimators=[("tsf", TSFRegressor(n_estimators=2), "all")]
    )
    ens.fit(X, y)

    assert len(ens.estimators_[0][2]) == 3
    assert ens.predict(X).shape == (X.shape[0],)


def test_single_estimator_split():
    """Test that a single split estimator correctly creates an estimator per channel."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(
        estimators=("tsf", TSFClassifier(n_estimators=2), "all-split")
    )
    ens.fit(X, y)

    assert len(ens.estimators_) == 3
    assert isinstance(ens.estimators_[0][2], int)
    assert ens.predict(X).shape == (X.shape[0],)

    ens = ChannelEnsembleRegressor(
        estimators=("tsf", TSFRegressor(n_estimators=2), "all-split")
    )
    ens.fit(X, y)

    assert len(ens.estimators_) == 3
    assert isinstance(ens.estimators_[0][2], int)
    assert ens.predict(X).shape == (X.shape[0],)


def test_remainder():
    """Test that the remainder is applied to remaining channels."""
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(
        estimators=[("tsf", TSFClassifier(n_estimators=2), 0)],
        remainder=TSFClassifier(n_estimators=2),
    )
    ens.fit(X, y)

    assert len(ens._remainder[2]) == 2
    assert ens.predict(X).shape == (X.shape[0],)

    ens = ChannelEnsembleRegressor(
        estimators=[("tsf", TSFRegressor(n_estimators=2), 0)],
        remainder=TSFRegressor(n_estimators=2),
    )
    ens.fit(X, y)

    assert len(ens._remainder[2]) == 2
    assert ens.predict(X).shape == (X.shape[0],)


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
