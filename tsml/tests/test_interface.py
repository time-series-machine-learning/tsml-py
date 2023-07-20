"""Unit tests for tsml interface."""
import numpy as np
import pytest

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.testing import (
    generate_2d_test_data,
    generate_3d_test_data,
    generate_unequal_test_data,
)


def _generate_conversion_test_X(data_type):
    if data_type == "3darray":
        X = generate_3d_test_data(n_channels=2)[0]
        return X, X.shape
    elif data_type == "2darray":
        X = generate_2d_test_data()[0]
        return X, (X.shape[0], 1, X.shape[1])
    elif data_type == "np_list":
        X = generate_unequal_test_data(n_channels=2)[0]
        return X, (len(X), 2, max([x.shape[1] for x in X]))
    else:
        raise ValueError(f"Invalid data_type: {data_type}")


@pytest.mark.parametrize("input_type", ("3darray", "2darray", "np_list"))
def test_convert_X_to_3d_array(input_type):
    est = _3dArrayDummy()
    X, old_shape = _generate_conversion_test_X(input_type)
    X = est._convert_X(X, pad_unequal=True)

    assert isinstance(X, np.ndarray)
    assert X.ndim == 3
    assert X.shape == old_shape

    est._validate_data(X)


@pytest.mark.parametrize("input_type", ("3darray", "2darray", "np_list"))
def test_convert_X_to_2d_array(input_type):
    est = _2dArrayDummy()
    X, old_shape = _generate_conversion_test_X(input_type)
    X = est._convert_X(X, concatenate_channels=True, pad_unequal=True)

    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.shape == (old_shape[0], old_shape[2] * old_shape[1])

    est._validate_data(X)


@pytest.mark.parametrize("input_type", ("3darray", "2darray", "np_list"))
def test_convert_X_to_numpy_list(input_type):
    est = _NpListDummy()
    X, old_shape = _generate_conversion_test_X(input_type)
    X = est._convert_X(X)

    assert isinstance(X, list)
    assert X[0].ndim == 2
    assert (len(X), X[0].shape[0], max([x.shape[1] for x in X])) == old_shape

    est._validate_data(X)


class _3dArrayDummy(BaseTimeSeriesEstimator):
    def __init__(self):
        super(_3dArrayDummy, self).__init__()

    def _more_tags(self) -> dict:
        return {"X_types": ["3darray"]}


class _2dArrayDummy(BaseTimeSeriesEstimator):
    def __init__(self):
        super(_2dArrayDummy, self).__init__()

    def _more_tags(self) -> dict:
        return {"X_types": ["2darray"]}


class _NpListDummy(BaseTimeSeriesEstimator):
    def __init__(self):
        super(_NpListDummy, self).__init__()

    def _more_tags(self) -> dict:
        return {"X_types": ["np_list"]}
