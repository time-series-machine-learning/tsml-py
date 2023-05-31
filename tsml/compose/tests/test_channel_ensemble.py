# -*- coding: utf-8 -*-
from tsml.compose._channel_ensemble import ChannelEnsembleClassifier
from tsml.interval_based import TSFClassifier
from tsml.utils.testing import generate_3d_test_data


def test_single_estimator():
    X, y = generate_3d_test_data(n_channels=3)

    ens = ChannelEnsembleClassifier(
        estimators=[("tsf", TSFClassifier(n_estimators=2), "all")]
    )
    ens.fit(X, y)

    assert len(ens.estimators_[0][2]) == 3
    assert ens.predict(X).shape == (X.shape[0],)
