# -*- coding: utf-8 -*-
import os

from numpy.testing import assert_array_almost_equal, assert_array_equal

from tsml.datasets import load_from_ts_file
from tsml.datasets.tests._expected_data_io_output import (
    equal_multivariate_X,
    equal_multivariate_y,
    equal_univariate_X,
    equal_univariate_y,
    unequal_multivariate_X,
    unequal_multivariate_y,
    unequal_univariate_X,
    unequal_univariate_y,
)


def test_load_from_ts_file_equal_univariate():
    """Load an equal length univariate time series from a file."""
    data_path = (
        "./tsml/datasets/MinimalChinatown/MinimalChinatown_TRAIN.ts"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../MinimalChinatown/MinimalChinatown_TRAIN.ts"
    )

    X, y = load_from_ts_file(data_path)

    assert_array_almost_equal(X, equal_univariate_X)
    assert_array_equal(y, equal_univariate_y)


def test_load_from_ts_file_unequal_univariate():
    """Load an unequal length univariate time series from a file."""
    data_path = (
        "./tsml/datasets/UnequalMinimalChinatown/UnequalMinimalChinatown_TRAIN.ts"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../UnequalMinimalChinatown/UnequalMinimalChinatown_TRAIN.ts"
    )

    X, y = load_from_ts_file(data_path)

    for i, x in enumerate(X):
        assert_array_almost_equal(x, unequal_univariate_X[i])
    assert_array_equal(y, unequal_univariate_y)


def test_load_from_ts_file_equal_multivariate():
    """Load an equal length multivariate time series from a file."""
    data_path = (
        "./tsml/datasets/EqualMinimalJapaneseVowels/EqualMinimalJapaneseVowels_TRAIN.ts"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../EqualMinimalJapaneseVowels/EqualMinimalJapaneseVowels_TRAIN.ts"
    )

    X, y = load_from_ts_file(data_path)

    assert_array_almost_equal(X, equal_multivariate_X)
    assert_array_equal(y, equal_multivariate_y)


def test_load_from_ts_file_unequal_multivariate():
    """Load an unequal length multivariate time series from a file."""
    data_path = (
        "./tsml/datasets/MinimalJapaneseVowels/MinimalJapaneseVowels_TRAIN.ts"
        if os.getcwd().split("\\")[-1] != "tests"
        else "../MinimalJapaneseVowels/MinimalJapaneseVowels_TRAIN.ts"
    )

    X, y = load_from_ts_file(data_path)

    for i, x in enumerate(X):
        assert_array_almost_equal(x, unequal_multivariate_X[i])
    assert_array_equal(y, unequal_multivariate_y)
