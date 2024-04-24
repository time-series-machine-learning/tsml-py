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
        random_state=0,
    )
    rotf.fit(X, y)

    assert len(rotf.estimators_) > 1

    # test train estimate
    proba = rotf.predict_proba(X)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(X), 2)
