# -*- coding: utf-8 -*-
"""Unit tests for sklearn classifiers."""

__author__ = ["MatthewMiddlehurst"]


from tsml.utils.testing import generate_test_estimators, parametrize_with_checks


@parametrize_with_checks(generate_test_estimators())
def test_check_estimator(estimator, check):
    """Test that sklearn estimators adhere to sklearn conventions."""
    check(estimator)
