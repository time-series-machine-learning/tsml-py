"""Rotation Forest test code."""

__author__ = ["MatthewMiddlehurst"]

import numpy as np

from tsml.datasets import load_minimal_chinatown
from tsml.vector import RotationForestClassifier


def test_contracted_rotf():
    """Test of RotF contracting and train estimate on unit test data."""
    # load unit test data
    X, y = load_minimal_chinatown(split="train")
    X = np.reshape(X, (X.shape[0], -1))

    rotf = RotationForestClassifier(
        contract_max_n_estimators=5,
        time_limit_in_minutes=0.25,
        save_transformed_data=True,
        random_state=0,
    )
    rotf.fit(X, y)

    assert len(rotf.estimators_) > 1

    # test train estimate
    train_proba = rotf._get_train_probs(X, y)
    assert isinstance(train_proba, np.ndarray)
    assert train_proba.shape == (len(X), 2)
    np.testing.assert_almost_equal(train_proba.sum(axis=1), 1, decimal=4)
