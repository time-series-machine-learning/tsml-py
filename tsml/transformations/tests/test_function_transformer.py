# -*- coding: utf-8 -*-
from tsml.transformations import FunctionTransformer
from tsml.utils.numba_functions.general import first_order_differences_3d
from tsml.utils.numba_functions.stats import row_mean
from tsml.utils.testing import generate_2d_test_data, generate_3d_test_data


def test_function_2d_mean():
    X, y = generate_2d_test_data()

    func = FunctionTransformer(func=row_mean)

    assert func.fit_transform(X).shape == (X.shape[0],)


def test_function_3d_diff():
    X, y = generate_3d_test_data()

    func = FunctionTransformer(func=first_order_differences_3d)

    assert func.fit_transform(X).shape == (X.shape[0], X.shape[1], X.shape[2] - 1)
