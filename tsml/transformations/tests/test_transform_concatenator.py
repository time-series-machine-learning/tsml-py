# -*- coding: utf-8 -*-
from tsml.transformations import (
    Catch22Transformer,
    FunctionTransformer,
    SevenNumberSummaryTransformer,
    TransformerConcatenator,
)
from tsml.utils.numba_functions.general import first_order_differences_3d
from tsml.utils.testing import generate_3d_test_data


def test_concatenate_features():
    X, y = generate_3d_test_data()

    concat = TransformerConcatenator(
        transformers=[
            Catch22Transformer(features=["DN_HistogramMode_5", "DN_HistogramMode_10"]),
            SevenNumberSummaryTransformer(),
        ]
    )

    assert concat.fit_transform(X).shape == (X.shape[0], 9)


def test_concatenate_series():
    X, y = generate_3d_test_data()

    concat = TransformerConcatenator(
        transformers=[
            FunctionTransformer(func=first_order_differences_3d),
            FunctionTransformer(func=first_order_differences_3d),
        ]
    )

    assert concat.fit_transform(X).shape == (X.shape[0], 1, 22)
