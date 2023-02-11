# -*- coding: utf-8 -*-
"""Unit tests for sklearn classifiers."""

__author__ = ["MatthewMiddlehurst"]

from sklearn.utils.estimator_checks import parametrize_with_checks

from tsml.sklearn import CITClassifier, RotationForestClassifier


@parametrize_with_checks([RotationForestClassifier(n_estimators=3), CITClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    """Test that sklearn estimators adhere to sklearn conventions."""
    check(estimator)
