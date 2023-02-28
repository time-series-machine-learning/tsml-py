# -*- coding: utf-8 -*-
"""Unit tests for tsml classifiers."""

__author__ = ["MatthewMiddlehurst"]


from tsml.utils.testing import generate_test_estimators, parametrize_with_checks


@parametrize_with_checks(generate_test_estimators())
def test_check_estimator(estimator, check):
    """Test that tsml and sklearn estimators adhere to conventions."""
    check(estimator)
